# GLM-5.3-Flash · Maya-S v2 IQ2_XXS — WebGL physical weight atlas + live telemetry

A zero-dependency (stdlib Python, no-build JS) telemetry stack for
**modified-maya** serving **GLM-5.3-Flash** (the `Maya-S-v2-IQ2_XXS`
3-shard GGUF pack + `DFlash2-Q4_K_M` drafter) on a unified-memory
**NVIDIA GB10** APU. Adapted from the
`strata-qwen38fn-unslothGGUF-RTX-telemetry` approach — same idea, GLM-shaped.

The scene: **46 plates in an 8×6 grid** (42 main MoE blocks + dense-lead/DSA
blocks + 1 NextN draft plate), each MoE block showing a **24×12 grid = 288
expert cells** (12,096 expert slots total, top-8 routing). Rims: cyan = MLA
attention, magenta = DSA (sparse indexer), green = NextN draft. Gold token
particles travel the embed → 46 plates → head path; amber flash marks expert
cells routed on the current token (see honesty note below). Front-left, a
32-cell KV growth bar (1 cell = capacity/32 tokens; green = VRAM-resident
prefix, blue = KV streamed from RAM).

## Layout

    tools/build_atlas.py     GGUF / pack index.txt / STRP profile -> web/atlas_data.js
                             (window.ATLAS_META + window.ATLAS_HEAT) and
                             live/atlas_manifest.json
    collect/collector.py     polls http://127.0.0.1:8009/metrics (token `sk-local`,
                             override with MAYA_METRICS_TOKEN), tails the maya log,
                             atomically writes live/telemetry.json, serves web/
                             statically (/api/telemetry, /api/atlas); --bind
                             tailscale re-binds the HTTP side to the tailscale0
                             address (metrics poll always stays on 127.0.0.1)
    collect/log_parser.py    regex event taxonomy for the maya/glm dialect
                             (Strata dialect kept parseable as legacy); unmatched
                             lines are kept as {"type":"log"} with the raw text
    collect/llm-scope.py     standalone read-only log dashboard
    web/atlas_gl.js          WebGL2 renderer (plates, expert grids, token
                             particles, KV growth bar, DFlash2 draft loop)
    web/index.html           DOM HUD: phase, tok/s, prompt/out, DFlash2 accept,
                             expert residency (VRAM + RAM tiers), tier traffic,
                             hardware, KV, spend saved, event log, sparklines
                             (y-axis legends + live current value)
    test/test_all.py         unittest suite: GGUF decode, STRP profile semantics,
                             native_experts arithmetic, log dialect, compose()
    deploy/maya-glm-atlas.service   systemd user unit (serve on :8091, bound to the tailscale0 interface)

See **TECHNIQUES.md** for the deep dive: the maya GGUF writer quirk, STRP
profile semantics, the log-dialect census, instanced-quad rendering, and how
the render loop was verified headlessly.

## Run

    python3 tools/build_atlas.py                       # rebuild atlas data
    python3 collect/collector.py --serve-port 8091     # loopback only
    python3 collect/collector.py --serve-port 8091 --bind tailscale

The atlas server needs the model engine running (maya on `127.0.0.1:8009`,
`/metrics` with `Authorization: Bearer sk-local`). With `--bind tailscale`,
open `http://<tailscale-ip>:8091/` from any tailnet device; the HUD fetches
`/api/telemetry` and `/api/atlas` relative to the page URL, so it just works.

    python3 -m unittest discover -s test -v           # 13 tests, all pass

## What the atlas shows

- **Heat colour** per expert cell: rank-ordered expert-residency heat from the
  STRP profile (`expert-profile-glm.bin`, 13,248 slots), navy → teal → gold.
- **Amber flash**: which experts are routed for the token currently in flight
  (synthetic overlay — see honesty note).
- **Forward-pass strip** (top bar): the stage the token head is at
  (`EMB → …` per-block stages → `HEAD`), recomputed per frame from the
  particle position; per-block arrival glow as a soft amber sheet.
- **DFlash2 draft loop**: green particles orbiting the DRAFT slab while
  speculation is active (`live.speculation == ["DFlash2"]`).
- **KV growth bar**: `kv.seq_tokens` against the 262,144-token served context
  (int8 KV; 1,048,576 trained).
- **HUD numbers** (all real, from `/metrics` + log tail): decode tok/s,
  prompt/output totals, DFlash2 accept/offered, expert residency
  (VRAM slots + RAM tier), tier traffic (RAM fetch, disk read, promotions),
  GPU util/temp/CPU, VRAM pool, storage, USD saved vs cloud, event log.

## Live load test — 4 concurrent sessions, 40 seconds

Four concurrent streaming chat sessions (`/v1/chat/completions`, model
`glm-5.3-flash`, max_tokens 1200, thinking on) hammered the engine while the
atlas server was sampled every 2 s. All four sessions ran without errors
(558 stream chunks received). The engine multiplexes them: one generation
active at a time (`active: 1`), the rest queued — so decode speed stays in
the normal single-session band while the expert cache warms visibly.

Captured telemetry (all values real, from `/api/telemetry`; RAM-fetch and
disk-read are GB/s tier traffic):

| t (s) | phase | tok/s | DFlash2 accept | expert VRAM hit | RAM fetch | disk | GPU % | °C | KV seq |
|---|---|---|---|---|---|---|---|---|---|
|  0 | writing a tool call: bas | 15.7 | 91.6 | 97.2 | 0.08 | 9.0 | 95 | 77 | 44,470 |
|  2 | reading the prompt | – | 91.3 | 97.2 | 0.08 | 9.1 | 94 | 77 | 45,752 |
|  4 | thinking | 9.3 | 91.3 | 97.2 | 0.08 | 9.1 | 71 | 79 | 45,752 |
|  6 | thinking | 11.6 | 94.7 | 94.1 | 0.28 | 22.2 | 95 | 75 | 45,752 |
|  8 | thinking | 11.5 | 83.3 | 95.2 | 0.27 | 18.4 | 95 | 77 | 45,752 |
| 10 | thinking | 11.5 | 80.6 | 95.5 | 0.34 | 16.3 | 95 | 76 | 45,752 |
| 12 | thinking | 11.8 | 80.5 | 96.0 | 0.31 | 14.5 | 95 | 77 | 45,752 |
| 14 | thinking | 12.2 | 80.4 | 96.3 | 0.30 | 13.2 | 95 | 77 | 45,752 |
| 16 | thinking | 13.9 | 85.0 | 96.8 | 0.27 | 11.5 | 94 | 77 | 45,752 |
| 18 | thinking | 15.2 | 83.1 | 97.0 | 0.24 | 10.7 | 95 | 77 | 45,752 |
| 20 | thinking | 15.9 | 85.4 | 97.3 | 0.22 | 9.4 | 94 | 77 | 45,752 |
| 22 | thinking | 15.7 | 87.1 | 97.4 | 0.22 | 9.0 | 95 | 78 | 45,752 |
| 24 | thinking | 14.8 | 88.4 | 97.6 | 0.21 | 8.4 | 93 | 77 | 45,752 |
| 26 | thinking | 14.8 | 89.3 | 97.7 | 0.20 | 7.9 | 95 | 77 | 45,752 |
| 28 | thinking | 15.2 | 88.7 | 97.8 | 0.18 | 7.7 | 95 | 78 | 45,752 |
| 30 | thinking | 16.6 | 89.9 | 97.9 | 0.17 | 7.1 | 94 | 78 | 45,752 |
| 32 | thinking | 15.4 | 89.7 | 98.0 | 0.15 | 6.9 | 95 | 78 | 45,752 |
| 34 | thinking | 16.0 | 89.0 | 98.1 | 0.14 | 6.6 | 95 | 78 | 45,752 |
| 36 | thinking | 14.8 | 88.9 | 98.1 | 0.14 | 6.4 | 95 | 77 | 45,752 |
| 38 | thinking | 15.6 | 88.7 | 98.2 | 0.14 | 6.2 | 94 | 78 | 45,752 |

Summary over the 40 s window:

- decode tok/s: **9.3 – 16.6** (mean 14.1)
- DFlash2 accept rate: **80.4 % – 94.7 %** (mean 87.3 %)
- expert VRAM hit: **94.1 % – 98.2 %** (mean 97.0 %) — climbing steadily as
  the four sessions' expert sets merge into the cache (disk reads decay
  22.2 → 6.2 GB/s over the window)
- prompt hit rate: **90.8 %** (stable — shared system prompt across sessions)
- GPU: **71 – 95 %** util, 75 – 79 °C, CPU 11 – 23 %
- KV sequence grew 44,470 → 45,752 tokens during the window
- engine phases observed: `writing a tool call: bash`, `reading the prompt`,
  `thinking`

To reproduce:

    python3 /tmp/load_test.py     # or any 4-worker load; sample /api/telemetry

## Honesty note (synthetic overlay)

The maya API exposes no per-token expert-routing ids, so the amber flash +
particle routing is a **synthetic overlay**: which experts light up is driven
by the STRP heat profile and decode activity, not by real routing decisions.
Everything else (heat, residency, tier traffic, KV, tok/s, accept rate,
hardware) is real measured data. The overlay is labelled as such in the HUD
legend and tooltip.

## Tests

    python3 -m unittest discover -s test -v

13 tests, stdlib-only: synthetic maya-style GGUF decode (u32 KV-type quirk),
STRP rank-ordered perfect matching, native_experts blob arithmetic, real-log
replay (>12 event types), `compose()` shape (per-token structural experts,
slots, MTP accept from request counters, GPU/pool), NextN blk.45 inventory,
shard None-tolerance.
