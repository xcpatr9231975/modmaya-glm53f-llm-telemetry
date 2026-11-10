# GLM-5.3-Flash · Maya-S v2 IQ2_XXS — WebGL physical weight atlas + live telemetry

A zero-dependency (stdlib Python, no-build JS) telemetry stack for
**modified-maya** serving **GLM-5.3-Flash** (the `Maya-S-v2-IQ2_XXS`
3-shard GGUF pack + `DFlash2-Q4_K_M` drafter) on a unified-memory
**NVIDIA GB10** APU. Adapted from the
`strata-qwen38fn-unslothGGUF-RTX-telemetry` approach — same idea, GLM-shaped:

46 plates in an 8×6 grid (44 main blocks + 3 dense-lead DSA + 1 NextN draft
plate), each MoE block showing a **24×12 grid = 288 expert cells**. Rims:
cyan = MLA attention, magenta = DSA (sparse indexer), green = NextN draft.
Amber flash = expert routed this token (see honesty note below).

## Layout

    tools/build_atlas.py     GGUF / pack index.txt / STRP profile -> web/atlas_data.js
                             (window.ATLAS_META + window.ATLAS_HEAT) and
                             live/atlas_manifest.json
    collect/collector.py     polls http://127.0.0.1:8009/metrics (token `sk-local`,
                             override with MAYA_METRICS_TOKEN), tails the maya log,
                             atomically writes live/telemetry.json, serves web/
                             statically (/api/telemetry, /api/atlas)
    collect/log_parser.py    regex event taxonomy for the maya/glm dialect
                             (Strata dialect kept parseable as legacy); unmatched
                             lines are kept as {"type":"log"} with the raw text
    collect/llm-scope.py     standalone read-only log dashboard
    web/atlas_gl.js          WebGL2 renderer (plates, expert grids, token
                             particles, KV growth bar, DFlash2 draft loop)
    web/index.html           DOM HUD: phase, tok/s, prompt/out, DFlash2 accept,
                             expert residency (VRAM + RAM tiers), tier traffic,
                             hardware, KV, spend saved, event log, sparklines
    test/test_all.py         unittest suite: GGUF decode, STRP profile semantics,
                             native_experts arithmetic, log dialect, compose()
    deploy/maya-glm-atlas.service   systemd user unit (serve on :8091, bound to the tailscale0 interface)

## Run

    python3 tools/build_atlas.py                  # rebuild atlas data
    python3 collect/collector.py --serve-port 8091   # loopback only
    python3 collect/collector.py --serve-port 8091 --bind tailscale
    #   -> serves on the tailscale0 IPv4 (e.g. http://100.81.58.85:8091/),
    #      reachable from any device on your tailnet
    python3 collect/collector.py --serve-port 8091 --bind 0.0.0.0
    #   -> all interfaces (LAN too)

Model artefacts (defaults, override with CLI flags):
`~/projects/Maya-data/models/Maya-S-v2-IQ2_XXS/*.gguf`,
`pack/index.txt`, `pack/native_experts.txt`, and the STRP routing profile
`~/projects/maya/data/expert-profile-glm.bin` (13,248 (layer, expert) rows).

## What the atlas shows

- **GGUF-derived**: all 1,412 tensors across 3 shards, quant types decoded
  from the maya writer's u32-KV-type quirk, per-expert gate/up/down byte
  regions from `native_experts.txt` (absolute shard offsets), vocab
  154,880 × hidden 4096, KV int8 with 262,144 served / 1,048,576 trained.
- **STRP profile**: rows are a **rank-ordered perfect matching** — the hot
  half pairs every expert slot exactly once into co-routing pairs ranked by
  routing co-occurrence (rank 0 = `(blk 14, expert 219) ↔ (blk 23, expert
  133)`); the cold tail pairs every global expert id once more as
  anti-co-routing. `ATLAS_HEAT` is the normalized pair rank.
- **Live telemetry**: expert residency (VRAM ~10,012 slots + RAM 63 slots
  of 12,096 total, ~68 GB pool), VRAM-hit / RAM-fetch / promotion traffic,
  DFlash2 accept rate, KV occupancy curve, NVML-measured GPU map.

## Honesty note (synthetic overlay)

The maya API exposes **no per-token expert ids**. The per-token amber
routing flicker and the "top-8 lit cells" are a *synthetic* overlay driven
by decode activity and the profile heat — labelled as such in the HUD.
Everything else (residency, tier traffic, accept rates, KV) is measured.

## Tests

    python3 -m unittest discover -s test -v
