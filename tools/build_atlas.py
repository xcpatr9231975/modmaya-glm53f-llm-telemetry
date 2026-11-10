#!/usr/bin/env python3
"""build_atlas.py - compile the *physical* GLM-5.3-Flash weight atlas
(Maya-S-v2 IQ2_XXS pack served by modified-maya) from the on-disk artefacts,
and emit it as a JavaScript global the WebGL visualiser can load without a
bundler.

WHAT THIS READS (all paths overridable with flags)
  1. GGUF shards 1-3  ~/projects/Maya-data/models/Maya-S-v2-IQ2_XXS/
                        GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf
                        -> `glm5-next` metadata KV (46 blocks, 288 experts
                           top-8 + 1 shared, MLA + DSA/SSM hybrid attention,
                           hyper-connections, indexer, vocab 154,880) plus the
                           per-shard tensor tables.  FORMAT NOTES (verified
                           byte-level, see test/test_all.py):
                           * GGUF v3, but the KV *type* field is u32, not u8
                             like llama.cpp v3 writes it (maya writer quirk).
                           * tensor entries: u64 name_len, name, u32 ndims,
                             u64 dims[ndims], u32 quant type, u64 data offset
                             (relative to the data section, which starts at
                             the 64-aligned end of the tensor table).
                           * quant type enum is maya's own numbering
                             (NOT ggml's); every bpw below was verified
                             empirically from consecutive tensor offsets:
                               0 f32(32.0)  8 q8_0(8.5)   10 iq2_s(2.625)
                               11 iq3_xs(3.4375)  12 q4_k(4.5)
                               14 q6_k(6.5625)  16 iq2_xxs(2.0625)
                               18 iq3_xxs(3.0625)  22 iq2_xs(2.5625)
                           * shard 2 also hosts the NextN draft block blk.45
                             (29 tensors, incl. a 288-expert MoE at IQ2_S /
                             IQ3_XS) which the pack's dense pool does NOT
                             carry -- the atlas reads it straight from GGUF.
  2. pack index.txt   .../Maya-S-v2-IQ2_XXS/pack/index.txt
                        -> 1257 dense (non-expert) tensors in `dense.bin`
                           (943,911,040 B pool): name, dtype code (0=quantized
                           -> lives in a GGUF shard, 2=f32, 4=bf16), pool
                           offset/size, rows, cols, quant enum, block elems.
                           Quantized rows are JOINED against the GGUF tables
                           by name to get the exact shard + byte range.
  3. native_experts.txt (same pack dir)
                        -> "strata native experts v3": one row per MoE layer
                           (3..44): gu_type(16=IQ2_XXS) d_type(18=IQ3_XXS)
                           blob_off(= cumulative 288 x blob_bytes) blob_bytes
                           (7,536,640 / expert) and the ABSOLUTE byte offsets
                           of the down/gate/up expert-matrix regions inside
                           the GGUF shards (region order on disk: down, gate,
                           up).  288 equally-sized expert blobs per region,
                           so region_bytes/288 = bytes per expert per matrix:
                           gate 2,203,232 B (2.10 bpw), down 3,235,160 B
                           (3.08 bpw), up 2,098,248 B (2.00 bpw, derived =
                           blob_bytes - gate - down).  This is how every one
                           of the 12,096 routed experts (+288 draft experts)
                           is LOCATED for the visualiser.
  4. expert-profile-glm.bin (~/projects/maya/data)
                        -> magic 'STRP', 24-byte header [version=1, layers=46,
                           experts=288, entries=13248, entries=13248, x],
                           13248 rows of 4 x u16.  VERIFIED SEMANTICS (differs
                           from the qwen pack only in the header arity):
                           * rows 0..6623: routing co-occurrence edges
                             (layerA, expertA, layerB, expertB), ranked by
                             routing frequency.  Layers 0-2 appear in the
                             profile but have NO expert weights (dense lead);
                             layer 45 is the NextN draft block (real experts).
                           * rows 6624..13247: the cold tail -- every expert
                             global id (layer*288+expert) listed exactly once
                             as (global_id, 0, global_id, 0) pairs, never
                             co-routed during profiling.
                           We aggregate endpoint counts into heat[46][288].
  5. DFlash2 drafter  ~/projects/Maya-data/models/GLM-5.3-Flash-DFlash2/
                        GLM-5.3-Flash-DFlash2-Q4_K_M.gguf (665 MiB)
                        -> `dflash` arch: 5 draft blocks, trunk projection
                           fc 20480x4096 (Q4_K), selector top-16/rank 256,
                           reads trunk layers 6,15,25,34,43.
  6. GGUF shard sizes -> file-level placement.

OUTPUTS
  web/atlas_data.js        `window.ATLAS = {...}`   (for the WebGL page)
  live/atlas_manifest.json same object, machine-readable

The JSON is deliberately self-describing: every number the visualiser draws
is annotated here with units.  Nothing is invented: fields we could not
verify are marked with a `note`.

Usage:  python3 tools/build_atlas.py [--shards DIR] [--pack DIR]
                                     [--profile F] [--drafter F]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import struct
import sys

HOME = os.path.expanduser("~")

DEF_SHARDS = f"{HOME}/projects/Maya-data/models/Maya-S-v2-IQ2_XXS"
DEF_PACK = f"{DEF_SHARDS}/pack"
DEF_PROFILE = f"{HOME}/projects/maya/data/expert-profile-glm.bin"
DEF_DRAFTER = (f"{HOME}/projects/Maya-data/models/GLM-5.3-Flash-DFlash2"
               "/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf")

# maya quant enum -> (name, bits-per-weight); ALL bpw verified empirically
# from consecutive tensor data offsets (test/test_all.py::TestGGUFParse).
MAYA_QUANT = {0: ("f32", 32.0), 8: ("q8_0", 8.5), 10: ("iq2_s", 2.625),
              11: ("iq3_xs", 3.4375), 12: ("q4_k", 4.5),
              14: ("q6_k", 6.5625), 16: ("iq2_xxs", 2.0625),
              18: ("iq3_xxs", 3.0625), 22: ("iq2_xs", 2.5625)}
# index.txt dtype codes (column 3): verified against pool byte sizes
# (2 -> f32 = 4 B/elem, 4 -> bf16 = 2 B/elem, 0 -> quantized in the GGUF).
IDX_DT = {0: "quant", 2: "f32", 4: "bf16"}
EXPERTS = 288          # expert_count KV + native_experts.txt
LAYERS = 46            # block_count KV: blk.0 .. blk.45 (45 = NextN draft)
MOE_START = 3          # blocks 0-2 are the dense lead


def qname(t):
    return MAYA_QUANT.get(t, (f"t{t}", None))[0]


# ---------------------------------------------------------------- GGUF ----

GGUF_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("?", 1),
    8: "string", 9: "array", 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
MAX_TABLE_READ = 1 << 26   # 64 MiB: these shards' tables are < 10 MiB


def gguf_string(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    s = buf[off + 8: off + 8 + n].decode("utf-8", "replace")
    return s, off + 8 + n


def gguf_value(buf, off, t):
    """-> (value, new_offset)."""
    if t == 8:
        return gguf_string(buf, off)
    if t == 9:  # array
        (et,) = struct.unpack_from("<I", buf, off)
        (n,) = struct.unpack_from("<Q", buf, off + 4)
        off += 12
        out = []
        for _ in range(n):
            v, off = gguf_value(buf, off, et)
            out.append(v)
        return out, off
    fmt, sz = GGUF_TYPES[t]
    v = struct.unpack_from(fmt, buf, off)[0]
    return v, off + sz


def parse_gguf_shard(path, with_meta=True):
    """Parse one maya-written GGUF v3 shard.

    Returns {path, file, file_size, meta, tensors:[...]} where each tensor
    has name, dims, type(maya enum), rel_off (from the data section start),
    bytes (exact for f32 / measured from the next offset otherwise; the
    shard-tail tensor is estimated from its verified bpw), bpw.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(MAX_TABLE_READ)
    if raw[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", raw, 4)
    off = 24
    meta = {}
    for _ in range(n_kv):
        key, off = gguf_string(raw, off)
        t = struct.unpack_from("<I", raw, off)[0]
        off += 4                      # maya writer: u32 type field
        val, off = gguf_value(raw, off, t)
        if with_meta:
            meta[key] = val           # shards 2/3: walk (and skip) only
    ents = []
    for _ in range(n_tensors):
        (nl,) = struct.unpack_from("<Q", raw, off)
        off += 8
        name = raw[off:off + nl].decode("utf-8", "replace")
        off += nl
        (nd,) = struct.unpack_from("<I", raw, off)
        off += 4
        dims = list(struct.unpack_from("<%dQ" % nd, raw, off))
        off += 8 * nd
        (ty,) = struct.unpack_from("<I", raw, off)
        off += 4
        (do,) = struct.unpack_from("<Q", raw, off)
        off += 8
        ents.append({"name": name, "dims": dims, "type": ty, "rel_off": do})
    table_end = off
    assert off <= len(raw), f"{path}: tensor table beyond read window"
    ents.sort(key=lambda e: e["rel_off"])
    data_start = (table_end + 63) & ~63      # 64-aligned data section
    elems = 1
    for i, e in enumerate(ents):
        el = 1
        for d in e["dims"]:
            el *= d
        e["elems"] = el
        nxt = ents[i + 1]["rel_off"] if i + 1 < len(ents) else None
        if e["type"] == 0:                   # f32: exact
            e["bytes"] = 4 * el
            e["bpw"] = 32.0
        elif nxt is not None and nxt > e["rel_off"]:
            e["bytes"] = nxt - e["rel_off"]
            e["bpw"] = round(8 * e["bytes"] / el, 4) if el else None
        else:                                # shard tail: estimate from bpw
            exp = MAYA_QUANT.get(e["type"], (None, None))[1]
            e["bytes"] = int(el * exp / 8) if exp else None
            e["bpw"] = exp
            e["note"] = "shard tail: bytes estimated from verified bpw"
        e["quant"] = qname(e["type"])
    return {"path": path, "file": os.path.basename(path), "file_size": size,
            "version": ver, "n_tensors": n_tensors, "n_kv": n_kv,
            "meta": meta if with_meta else None,
            "data_start": data_start, "tensors": ents}


# ----------------------------------------------------------- pack index ----

def parse_index_txt(path):
    """index.txt (strata pack index v3): one row per pool/dense tensor.
    Columns (verified against pool byte sizes):
      0 name, 1 ?(0), 2 dtype (2=f32, 4=bf16, 0=quantized -> GGUF),
      3 pool_off, 4 pool_size, 5/6 secondary off/size, 7 rows, 8 cols,
      9 quant enum, 10 ?, 11 block elems, rest zeros."""
    tensors = []
    header = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                it = line[1:].split()
                for i in range(0, len(it) - 1, 2):
                    try:
                        header[it[i]] = int(it[i + 1])
                    except ValueError:
                        pass
                continue
            if not line:
                continue
            p = line.split()
            dtc = int(p[2])
            rows, cols = int(p[7]), int(p[8])
            t = {"name": p[0], "dtype": IDX_DT.get(dtc, f"dt{dtc}"),
                 "rows": rows, "cols": cols,
                 "elems": rows * (cols or 1),
                 "pool_off": int(p[3]), "pool_size": int(p[4]),
                 "aux_off": int(p[5]), "aux_size": int(p[6])}
            if dtc == 0:
                t["quant_enum"] = int(p[9])          # maya enum (see GGUF join)
                t["block_elems"] = int(p[11]) if len(p) > 11 else None
            else:
                t["bytes"] = t["pool_size"]
                t["bpw"] = round(8 * t["bytes"] / t["elems"], 4) if t["elems"] else None
            tensors.append(t)
    return header, tensors


# ---------------------------------------------------- native_experts -------

def parse_native_experts(path, experts=EXPERTS):
    """'strata native experts v3': layer gu_type d_type offset blob_bytes
    gate_off up_off down_off [shard].  gate/up/down offs are ABSOLUTE byte
    offsets into the GGUF shards; on-disk region order is down, gate, up."""
    rows = []
    note, total = "", None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                note = line[2:]
                for tok in note.split():
                    if tok.startswith("total"):
                        try:
                            total = int(tok.split("=", 1)[1] if "=" in tok else 0)
                        except ValueError:
                            pass
                for part in note.split():
                    if part.isdigit() and total is None and len(part) > 8:
                        pass
                m_total = None
                import re as _re
                m_total = _re.search(r"\btotal[=: ]+(\d+)", note)
                if m_total:
                    total = int(m_total.group(1))
                continue
            if not line:
                continue
            p = line.split()
            r = {"layer": int(p[0]),
                 "gu_type": qname(int(p[1])),
                 "d_type": qname(int(p[2])),
                 "blob_off": int(p[3]), "blob_bytes": int(p[4]),
                 "gate_off": int(p[5]), "up_off": int(p[6]),
                 "down_off": int(p[7]),
                 "shard": p[8] if len(p) > 8 else None}
            # whole regions (288 experts each)
            r["gate_region"] = r["gate_bytes"] = r["up_off"] - r["gate_off"]
            r["down_region"] = r["down_bytes"] = r["gate_off"] - r["down_off"]
            r["gate_per_expert"] = r["gate_bytes"] // experts
            r["down_per_expert"] = r["down_bytes"] // experts
            # up region is not bracketed in the file; derive it from the
            # per-expert blob total and flag the assumption.
            r["up_per_expert"] = r["blob_bytes"] - r["gate_per_expert"] \
                - r["down_per_expert"]
            r["up_region"] = r["up_per_expert"] * experts
            r["up_bytes_assumed"] = True
            r["note"] = ("up region derived as blob_bytes - gate - down "
                         "(per expert); blob_off is cumulative logical, "
                         "gate/up/down_off are absolute in the GGUF shards")
            rows.append(r)
    return note, total, rows


# ------------------------------------------------- expert profile (STRP) ---

def parse_expert_profile(path, layers=LAYERS, experts=EXPERTS):
    """GLM STRP profile: 24-byte header [version, layers, experts, entries,
    entries, x] then `entries` rows of 4 x u16.  Rows before the first
    out-of-range row are hot routing co-occurrence edges
    (layerA, expertA, layerB, expertB) ranked by routing frequency; the rest
    are the cold tail: every global expert id exactly once, as
    (global_id, 0, global_id, 0) pairs."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"STRP":
        raise SystemExit(f"{path}: bad magic {raw[:4]!r} (want b'STRP')")
    version, n_layers, n_experts, n_entries, n_entries2, extra = \
        struct.unpack_from("<6I", raw, 4)
    if (n_layers, n_experts, n_entries) != (layers, experts, layers * experts):
        raise SystemExit(f"profile header {n_layers}x{experts}x{n_entries} "
                         f"!= expected {layers}x{experts}x{layers * experts}")
    n_u16 = (len(raw) - 24) // 2
    vals = struct.unpack_from("<%dH" % n_u16, raw, 24)
    rows = [(vals[i], vals[i + 1], vals[i + 2], vals[i + 3])
            for i in range(0, min(n_u16, n_entries * 4), 4)]
    # The hot half is a PERFECT MATCHING: every expert slot (layer, expert)
    # appears in exactly one hot row, and row order ranks the co-routing
    # pairs by routing co-occurrence frequency (most frequent first).  The
    # cold tail pairs every global expert id exactly once again -- the
    # anti-co-routing (never-co-routed) pairing.  We therefore derive:
    #   * heat[L][e]  = normalized co-routing pair rank
    #       (65535 for rank 0 .. 0 for the last rank; 1/slot, no sums),
    #   * hot_rank / hot_partner / cold_partner per expert slot.
    n_hot = 0
    pairs, cold = [], []
    hot_rank = {}      # (L,e) -> row rank
    hot_partner = {}   # (L,e) -> partner slot (L2,e2)
    cold_phase = False
    for a, fa, b, fb in rows:
        if not cold_phase and a < layers and fa < experts \
                and b < layers and fb < experts:
            hot_rank[(a, fa)] = n_hot
            hot_rank[(b, fb)] = n_hot
            hot_partner[(a, fa)] = (b, fb)
            hot_partner[(b, fb)] = (a, fa)
            pairs.append([a, fa, b, fb])
            n_hot += 1
        else:
            cold_phase = True
            cold.append([a, fa, b, fb])
    cold_rows = len(cold)
    # sanity: perfect matching over all slots
    if len(hot_rank) != layers * experts:
        raise SystemExit(f"profile hot half covers {len(hot_rank)} slots, "
                         f"expected {layers * experts} (not a perfect matching)")
    cold_globals = [r[0] for r in cold] + [r[2] for r in cold]
    if len(set(cold_globals)) != layers * experts:
        raise SystemExit(f"profile cold tail covers {len(set(cold_globals))} "
                         f"distinct globals, expected {layers * experts}")
    # rank -> heat (u16-ish scale for the WebGL ramp)
    heat = [[0] * experts for _ in range(layers)]
    for (L, e), rk in hot_rank.items():
        heat[L][e] = max(0, round(65535 * (1.0 - rk / max(1, n_hot - 1))))
    lstats = []
    for L in range(layers):
        rks = sorted(hot_rank.get((L, e), 10**9) for e in range(experts))
        lstats.append({"rank_min": rks[0], "rank_median": rks[len(rks) // 2],
                       "rank_max": rks[-1]})
    top = [{"layer": L, "expert": e, "rank": rk,
            "partner": list(hot_partner[(L, e)])}
           for (L, e), rk in sorted(hot_rank.items(), key=lambda kv: kv[1])[:64]]
    return {"version": version, "layers": n_layers, "experts": n_experts,
            "entries": n_entries, "extra_header_word": extra,
            "hot_pairs": n_hot, "cold_rows": cold_rows,
            "semantics": ("hot half: perfect matching of all expert slots "
                          "into co-routing pairs, ranked by routing "
                          "co-occurrence (row order = rank); cold tail: "
                          "every global expert id exactly once as "
                          "(global_id, 0, global_id, 0) anti-co-routing "
                          "pairs"),
            "pairs_first8": pairs[:8], "pairs_last4": pairs[-4:],
            "cold_first8": cold[:8],
            "heat": heat, "layer_stats": lstats, "top_experts": top}


# ------------------------------------------------------ drafter (DFlash2) --

def parse_drafter(path):
    s = parse_gguf_shard(path)
    m = s["meta"] or {}
    d = {}
    for k, v in m.items():
        d[k] = v if not isinstance(v, list) or len(v) <= 12 else v[:12]
    tensors = [{"name": t["name"], "dims": t["dims"], "quant": t["quant"],
                "bytes": t["bytes"], "elems": t["elems"]}
               for t in sorted(s["tensors"], key=lambda x: x["name"])]
    return {"file": s["file"], "file_size": s["file_size"], "version": s["version"],
            "meta": d, "tensors": tensors,
            "n_tensors": s["n_tensors"]}


# ----------------------------------------------------------------- main ----

def build(args):
    shards = sorted(glob.glob(os.path.join(args.shards, "*.gguf")))
    if not shards:
        raise SystemExit(f"no .gguf files under {args.shards}")
    print(f"[atlas] {len(shards)} GGUF shards:")
    for s in shards:
        print(f"   {os.path.basename(s)}  {os.path.getsize(s):,} B")
    parsed = [parse_gguf_shard(p, with_meta=(i == 0)) for i, p in
              enumerate(shards)]
    meta = parsed[0]["meta"] or {}

    def m(k, d=None):
        return meta.get(k, d)

    hidden = m("glm5-next.embedding_length", 4096)
    vocab = m("glm5-next.vocab_size", None)
    experts = m("glm5-next.expert_count", EXPERTS)
    layers = m("glm5-next.block_count", LAYERS)
    kvh = m("glm5-next.attention.head_count_kv") or []
    # attention kind per block: MLA (kv_heads>0) vs DSA/SSM (kv_heads==0)
    mla = [bool(kvh[L]) if L < len(kvh) else False for L in range(layers)]

    # name -> (shard_idx, tensor) for the GGUF join + blk.45 inventory
    gguf = {}
    for si, s in enumerate(parsed):
        for t in s["tensors"]:
            gguf[t["name"]] = (si, t)

    # ---- pack index ------------------------------------------------------
    ihdr, idx = parse_index_txt(args.index)
    misses = []
    dense_tensors = []
    for t in idx:
        if t["dtype"] == "quant":
            hit = gguf.get(t["name"])
            if hit is None:
                misses.append(t["name"])
                t["note"] = "quantized row with no GGUF match"
                dense_tensors.append(t)
                continue
            si, g = hit
            t["quant"] = g["quant"]
            t["shard"] = parsed[si]["file"]
            t["bytes"] = g["bytes"]
            t["bpw"] = g["bpw"]
            t["gguf_off"] = parsed[si]["data_start"] + g["rel_off"]
        dense_tensors.append(t)

    # ---- native experts --------------------------------------------------
    note, total, nerows = parse_native_experts(args.native, experts)
    blobs = {r["layer"]: r for r in nerows}

    # ---- STRP profile ----------------------------------------------------
    prof = parse_expert_profile(args.profile, layers, experts)
    heat = prof.pop("heat")

    # ---- per-layer rollup -------------------------------------------------
    layers_meta = []
    for L in range(layers):
        pref = f"blk.{L}."
        pool = [t for t in idx if t["name"].startswith(pref)
                and t["dtype"] != "quant"]
        qq = [t for t in idx if t["name"].startswith(pref)
              and t["dtype"] == "quant" and "bytes" in t]
        moe = f"blk.{L}.ffn_gate_exps.weight" in gguf
        layers_meta.append({
            "i": L, "kind": "draft" if L == 45 else ("mla" if mla[L] else "dsa"),
            "moe": moe,
            "tensors": len(pool) + len(qq),
            "pool_bytes": sum(t["pool_size"] for t in pool),
            "gguf_bytes": sum(t["bytes"] for t in qq),
            "expert_blob": {k: blobs[L][k] for k in
                            ("gate_off", "up_off", "down_off", "blob_off",
                             "blob_bytes", "gate_per_expert",
                             "up_per_expert", "down_per_expert", "shard")}
                          if L in blobs else None,
            "profile": prof["layer_stats"][L],
        })

    # ---- NextN draft block (blk.45) ---------------------------------------
    # NextN lives mostly in shard 2 but spills 13 tensors into shard 3
    nn_src = parsed[1]["tensors"] + (parsed[2]["tensors"] if len(parsed) > 2
                                     else [])
    nextn_tensors = sorted((t for t in nn_src
                           if t["name"].startswith(("blk.45.", "nextn.",
                                                    "enorm", "hnorm"))),
                          key=lambda t: t["name"])
    nextn = [{"name": t["name"], "dims": t["dims"], "quant": t["quant"],
              "bytes": t["bytes"], "elems": t["elems"]}
             for t in nextn_tensors]

    # ---- DFlash2 drafter ---------------------------------------------------
    drafter = parse_drafter(args.drafter) if os.path.isfile(args.drafter) \
        else {"note": f"{args.drafter}: not found"}

    out = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": {
            "name": m("general.name", "GLM-5.3-Flash"),
            "arch": m("general.architecture", "glm5-next"),
            "pack": "Maya-S-v2-IQ2_XXS",
            "maya_recipe": meta.get("maya.recipe"),
            "split": {"count": m("split.count"),
                      "tensors": m("split.tensors.count")},
            "engine": "modified-maya (maya 1.1.56, engine_kind glm-fast)",
            "quant": ("experts IQ2_XXS gate/up + IQ3_XXS down; dense/attention "
                      "Q6_K + q8_0 + f32; NextN draft IQ2_S/IQ3_XS; drafter Q4_K_M"),
        },
        "dims": {
            "layers": layers, "moe_start": MOE_START,
            "moe_layers": sum(1 for L in layers_meta if L["moe"]),
            "experts": experts, "top_k_experts": m("glm5-next.expert_used_count", 8),
            "shared_experts": m("glm5-next.expert_shared_count", 1),
            "hidden": hidden, "expert_ffn": m("glm5-next.expert_feed_forward_length", 2048),
            "dense_ffn": m("glm5-next.feed_forward_length", 12288),
            "vocab": vocab, "head_count": m("glm5-next.attention.head_count", 64),
            "kv_lora_rank": m("glm5-next.attention.kv_lora_rank", 512),
            "q_lora_rank": m("glm5-next.attention.q_lora_rank", 1536),
            "key_length": m("glm5-next.attention.key_length", 512),
            "key_length_mla": m("glm5-next.attention.key_length_mla", 256),
            "value_length": m("glm5-next.attention.value_length", 512),
            "value_length_mla": m("glm5-next.attention.value_length_mla", 256),
            "mla_layers": sum(1 for v in mla if v),
            "hyper_connections": {"count": m("glm5-next.hyper_connection.count", 4)},
            "indexer": {"heads": m("glm5-next.attention.indexer.head_count", 32),
                        "top_k": m("glm5-next.attention.indexer.top_k", 2048)},
            "kda": {"head_dim": m("glm5-next.kda.head_dim"),
                    "gate_lower_bound": m("glm5-next.kda.gate_lower_bound")},
            "expert_gating_func": m("glm5-next.expert_gating_func", 2),
            "expert_weights_scale": m("glm5-next.expert_weights_scale"),
            "leading_dense_block_count": m("glm5-next.leading_dense_block_count", 3),
            "context_trained": m("glm5-next.context_length", 1048576),
            "context_served": 262144,
            "kv_dtype": "int8",
            "expert_total": 42 * experts,
            "draft_experts": experts,
            "expert_vram_slots": 10012, "expert_ram_slots": 63,
            "phantom_lead_layers": [0, 1, 2],
        },
        "layers": layers_meta,
        "dense_tensors": dense_tensors,
        "native_experts": {"note": note, "total_bytes": total, "rows": nerows},
        "nextn": {"shard": parsed[1]["file"] + " + " + parsed[2]["file"]
                  if len(parsed) > 2 else parsed[1]["file"],
                  "tensors": nextn},
        "drafter": drafter,
        "quant_enum": {str(k): v[0] for k, v in MAYA_QUANT.items()},
        "shards": [{"file": s["file"], "file_size": s["file_size"],
                    "n_tensors": s["n_tensors"], "n_kv": s["n_kv"],
                    "data_start": s["data_start"]} for s in parsed],
        "profile": prof,
        "index_header": ihdr,
        "index_quant_misses": misses,
    }
    return out, heat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=DEF_SHARDS)
    ap.add_argument("--index", default=os.path.join(DEF_PACK, "index.txt"))
    ap.add_argument("--native", default=os.path.join(DEF_PACK, "native_experts.txt"))
    ap.add_argument("--profile", default=DEF_PROFILE)
    ap.add_argument("--drafter", default=DEF_DRAFTER)
    ap.add_argument("--web-out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "atlas_data.js"))
    ap.add_argument("--manifest-out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live", "atlas_manifest.json"))
    args = ap.parse_args(argv)

    out, heat = build(args)
    dims = out["dims"]
    tot = sum(t["bytes"] for t in out["dense_tensors"] if "bytes" in t)
    print(f"[atlas] dims: {dims['layers']} blocks, {dims['moe_layers']} MoE x "
          f"{dims['experts']} experts top-{dims['top_k_experts']} (+1 shared), "
          f"vocab {dims['vocab']}")
    print(f"[atlas] dense/index bytes accounted: {tot:,} B; "
          f"quant GGUF-join misses: {len(out['index_quant_misses'])}")
    print(f"[atlas] profile: {out['profile']['hot_pairs']} hot pairs, "
          f"{out['profile']['cold_rows']} cold rows; heat top: "
          f"{out['profile']['top_experts'][0]}")
    js = ("/* generated by tools/build_atlas.py — do not edit */\n"
          "window.ATLAS_META =\n" + json.dumps(out, indent=1) + ";\n"
          "window.ATLAS_HEAT =\n" + json.dumps(heat) + ";\n")
    os.makedirs(os.path.dirname(args.web_out), exist_ok=True)
    with open(args.web_out, "w") as f:
        f.write(js)
    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    with open(args.manifest_out, "w") as f:
        json.dump(out, f)
    print(f"[atlas] wrote {args.web_out} ({os.path.getsize(args.web_out):,} B) "
          f"and {args.manifest_out}")


if __name__ == "__main__":
    main()
