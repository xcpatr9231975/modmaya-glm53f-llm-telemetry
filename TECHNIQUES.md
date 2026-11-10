# TECHNIQUES.md — how the GLM-5.3-Flash telemetry stack actually works

Companion to `README.md` (architecture + live capture). This file records the
techniques, decoding quirks and verification methods, in pipeline order.

## 1. Decoding the maya GGUF (the writer quirk)

The model is `GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf`
(3 shards, 1,412 tensors) plus `GLM-5.3-Flash-DFlash2-Q4_K_M.gguf` (drafter).

- Maya's GGUF writer is **not byte-identical to upstream `llama.cpp`**: it
  emits KV-type fields as **u32** (not the u32-in-some-versions/u8-in-others
  mix upstream writers use) and uses its own quant-type enum. The rule that
  always works: **walk KV entries by type code, never by assumed field
  widths**. The unit test builds a synthetic maya-style GGUF in memory and
  locks the quirk (`test_all.py::TestGguf`).
- Tensor counts and names come straight from the shard headers; `blk.45`
  (NextN draft) tensors carry a `nextn.` prefix and are read directly from
  shards 2+3 (29 tensors incl. `blk.45.nextn.eh_proj.weight`).
- Quant semantics are **sourced empirically**: instead of trusting a quant
  enum, measure bits-per-weight per quant type from `file_size / n_params`
  of known tensors, then classify every tensor by bpw. IQ2_XXS lands at
  ~2.06 bpw; 0 quant misses across 1,412 tensors.

## 2. STRP expert profile — rank-ordered perfect matchings

`expert-profile-glm.bin` is 13,248 rows × 4×u16. It is **not summed heat**:
each row is one rank-ordered perfect matching — for token-slot *r*, the
13,248 (layer, expert) pairs sorted by routing preference. Decoding it as a
flat list and re-aggregating per (layer, expert) yields the per-slot rank
distribution used for `ATLAS_HEAT` (46 × 288). The hottest single slot pair
in the shipped data: (layer 14, expert 219) ↔ (layer 23, expert 133).

`parse_expert_profile` is stateless: `tools/build_atlas.py` reads rows
independently, so any trailing partial row is dropped, not mis-parsed.

## 3. The maya log dialect (and the census)

The engine log (`maya.py` output) is a line-based dialect: `glm stat:`,
`glm pack:`, `glm fast:`, `glm spec:`, `glm slots:`, `[maya] done:`,
`[maya] prompt cap:` and friends. Method:

1. **Digit-normalization census**: replace every digit run with `N`, count
   line shapes, and you get the complete grammar (151,496-line log →
   ~dozen shapes, 99.991 % coverage; 13 legitimate `log` fallbacks).
2. Each shape becomes one compiled regex in `collect/log_parser.py`
   (~60 regexes; maya/glm first, the legacy Strata dialect kept secondary).
3. **Statelessness**: parsing is per-line, so optional lines (e.g. `glm
   pack:` continuations that only appear under VRAM pressure) can never
   suppress a completed request. Multiline stat-line continuations may fall
   to the `log` fallback rather than be force-matched.
4. Nothing is silently dropped: unmatched lines become
   `{"type":"log","raw":<truncated line>}`.

`[maya] raw:` lines contain full think transcripts — the parser keeps only
their length. Answer events total 1,580 in the reference log.

## 4. Collector semantics (`collect/collector.py`)

- Polls `/metrics` every 2 s (`Authorization: Bearer sk-local`); tails the
  log with an open seek handle (re-open on shrink; the log is 15.5 MB and
  binary-ish → always `grep -a`, never trust encoding).
- `compose()` derives: accept-rate from `tiers.now.draft_acc/draft_off`
  with request-counter fallback (`98/118` in the reference window); expert
  residency = 43 MoE layers × top-8 = 344 structural experts/token against
  12,096 slots; KV occupancy keeps a 180-entry deque of `seq_tokens`.
- **Atomic write**: `telemetry.json.tmp` + rename, so `/api/telemetry`
  never serves a torn file even while two collectors run.
- `--once` writes one snapshot and exits, even with `--serve-port`; the live
  server must run without it.
- `--bind tailscale` re-binds only the HTTP-serving side: `_resolve_bind`
  prefers `ip -4 -o addr show tailscale0`, falls back to `tailscale ip -4`,
  and refuses otherwise (use `--bind 0.0.0.0` or a literal address). The
  metrics poll stays on `127.0.0.1:8009`.

## 5. Building the atlas (`tools/build_atlas.py`)

- Grid derivation: `gridFor(n)` picks the nearest divisor pair to a 2:1
  aspect — 288 experts → **24×12** cells per plate (qwen's 512 → 32×16).
- Layout: 46 layers → 8-column × 6-row plate grid; plate = 40×26 world
  units, expert pitch `CELL=1.3, CELL_GAP=0.25` (scaled from qwen's
  `1.02/0.135` for 32 columns on the same 40-unit plate).
- Rim colour by attention kind: MLA cyan, DSA magenta, NextN draft green
  (`34 dsa / 11 mla / 1 draft`). The PLE gold notch is dropped (`ple: 0`
  in this pack).
- Output: `web/atlas_data.js` (`window.ATLAS_META` 46-block manifest +
  `window.ATLAS_HEAT` 46×288, ~545 KB) and `live/atlas_manifest.json`
  (1,412 tensors, 1,257 dense rows, 0 quant misses). `expert_blob` keys:
  `gate_per_expert / up_per_expert / down_per_expert / blob_bytes / shard`;
  `blob_bytes` is the **per-expert** sum and
  `up_per_expert = blob_bytes - gate - down` with `up_bytes_assumed: True`.

## 6. Rendering: one quad, instanced everything (`web/atlas_gl.js`)

The whole scene is **one quad geometry** drawn with `drawArraysInstanced`
over instance tuples of stride 32 bytes: `(off.xy, sz.xy, col.rgb, kind)`.
`kind` selects the fragment shader's shape path (0 rim, 1 plate, 2 expert
cell, 3 bar, 4 particle, 5 slab, 6 grid line).

- **Static batch** (`statBuf`): rims, plates, router bars, slabs.
- **Dynamic batch** (`dynBuf`, one instance per expert cell): colours are
  re-computed **every frame** (heat colour + amber flash decay) and shipped
  with one `bufferSubData` of the stride-8 array; offsets/sizes/kind are
  written once at build (`_patchDynStatic`).
- **Particles + glow** are pushed into a fresh buffer per frame and drawn in
  an **additive pass** (`blendFunc(SRC_ALPHA, ONE)`, depth mask off) —
  token particles along the embed→plates→head polyline, the DFlash2 orbit,
  and per-block amber arrival sheets (`layerAct`).
- Camera: orbit (drag), zoom (wheel), pan (right-drag), click-to-fly, `R`
  resets; picking intersects the y=0 plane and maps px/pz to
  (layer, expert) with exact grid algebra — `pickAt` uses `this.GRID.c/r`,
  the same constants as the draw path, so hover matches what you see.

## 7. HUD wiring (`web/index.html`)

- Poll chain `api/telemetry → ../live/telemetry.json → telemetry.json`
  every 1500 ms; fetch failures and render failures are caught **separately**
  so a broken scene can't hide live data, and a `netstat` label in the
  header shows poll state + data age.
- WebGL2-less fallback: if `new AtlasGL()` throws, the HUD keeps running
  (this happened — see §9).
- Sparklines draw with y-axis legends (dim max top-right, actual min
  bottom-right) and the **current value bold top-left** with a dot on the
  newest sample, compactly formatted (`262k`, `12.6`, `92.1`).
- The forward-pass stage chips read `gl.fwdStage`, recomputed every frame
  from the lead particle's position along the polyline (49 segments:
  embed, 46 plates, head).

## 8. Deployment

`deploy/maya-glm-atlas.service` (systemd **user** unit) runs the collector
with `--serve-port 8091 --bind tailscale --from-start`. Tailscale binding
was verified live: single listener on the tailscale0 IPv4, metrics poll
untouched on loopback, HUD reachable from any tailnet device.

## 9. Headless verification (how the dead-scene bug was found)

The 24×12-grid port dropped the `CELL`/`CELL_GAP` constants, so
`new AtlasGL` threw, the boot fallback swapped the canvas out, and the
render loop never started — telemetry HUD worked, scene was dead. Found by
running the real scripts under **Node's `vm` with a Proxy-based WebGL2
stub**: `get` returns `1` for uppercase constants, no-op functions for
methods, and truthy for `getShaderParameter`/`getProgramParameter`. Then:

- 120 simulated frames of `gl.frame(0.0167)` with real `/api/telemetry`
  data → caught `ReferenceError: CELL is not defined` on frame 1, and after
  the fix reported `particles: 26, flashCount: 1991, fwdStage: ROUTER
  softmax (layer 19)`.
- Same technique validated the HUD (`renderHud` with a stub DOM), the
  sparkline legends, and the poll-chain resilience.

Rule of thumb: **if the renderer is the only untestable piece, stub the GL
context — the JS is testable end-to-end without a browser.**

## 10. Honesty (repeated from README)

Per-token expert routing is not exposed by the maya API. The amber flash and
particle routing are a synthetic overlay driven by STRP heat + decode
activity; everything else is measured. Labelled as such in the HUD legend,
tooltip, README and here.
