"""log_parser.py — one engine-log line -> event(s) (dict | list | None).

Primary dialect: the modified-maya engine (GLM-5.3-Flash · Maya-S-v2-IQ2_XXS
on the NVIDIA GB10), written to ~/projects/maya/maya-maya-s-v2-iq2_xxs-256k.log
(`[maya] ...` / `glm ...` / `glm ...:` / `maya serve:` / `maya generate:` lines).

Secondary dialect: the Strata server (Qwen3.8-Flash-Next on the A5000) —
`strata serve: ...` lines — kept so the same collector still reads old logs.

Lines that match nothing are kept as type "log" with the raw line truncated,
so nothing is silently dropped.

Parsing is stateless; optional lines never suppress completed requests.
"""
from __future__ import annotations

import re

def _n(s):
    """'41,911' -> 41911; '12.3' -> 12.3; None on failure."""
    try:
        return float(s.replace(",", "")) if s else None
    except ValueError:
        return None

# ============================================================ maya / glm ===
# [maya] req openai: msgs=44 tools=80 max_tokens_raw=32000/None max_new=32000 thinking=True stream=True prompt_tokens=41912 last='user':'...'
RE_REQ = re.compile(
    r"\[maya\] req (\w+): msgs=(\d+) tools=(\d+) max_tokens_raw=(\S+) max_new=(\S+) "
    r"thinking=(\w+) stream=(\w+) prompt_tokens=([\d,]+) last='([^']+?)['\"]:([\s\S]*)")
# [maya] reading the prompt: 41,911 of 41,912 tokens, 3 s so far
RE_READ = re.compile(r"\[maya\] reading the prompt: ([\d,]+)(?: of ([\d,]+))? tokens, ([\d.]+) s so far")
# [maya] reading the prompt: 8,000 of max 32,000 tokens, 40 tok/s, 20 s
RE_READ2 = re.compile(r"\[maya\] reading the prompt: ([\d,]+) of max ([\d,]+) tokens, ([\d.]+) tok/s, ([\d.]+) s")
# glm prefill: CUDA0 chunk 1/1, 7766 tokens, wall 12312 ms | the copy lane 11397 ms: copies 2223 (7.4 GB/s), ring wait 2184, gaps 1, mixer windows 6988 | lead 817, tail 98 | pcie 16.49 GB (ram 0.00, disk 16.49) + 1.21 GB vram | experts: staged 2687, resident 9121 (2055979 rows), prestaged 59, cpu 0 (0 rows, 0 ms) | plan 1 ms host, disk wait 0 ms, hop in 0 ms | layer lane 134.9/75.6 ms max/min
RE_CHUNK = re.compile(
    r"glm prefill: (?:CUDA0|HIP) chunk (\d+)/(\d+), ([\d,]+) tokens, wall (\d+) ms \| the copy lane (\d+) ms: "
    r"copies (\d+) \(([\d.]+) GB/s\).*pcie ([\d.]+) GB \(ram ([\d.]+), disk ([\d.]+)\)(?: \+ ([\d.]+) GB vram)?"
    r".*experts: staged (\d+), resident (\d+)(?: \(([\d,]+) rows\))?(?:, prestaged (\d+))?"
    r"(?:, cpu (\d+))?")
# glm prefill: 461 tokens in 2252.6 ms (4.89 ms/token, 205 tok/s), 1 chunk of 512 tokens | CUDA0 2251 ms
RE_PREFILL_SUM = re.compile(
    r"glm prefill: ([\d,]+) tokens in ([\d.]+) ms \(([\d.]+) ms/token, ([\d.]+) tok/s\), (\d+) chunk(?:s)? of ([\d,]+) tokens")
# glm prefill: CUDA0 chunks of 32768 tokens (--prefill auto, up to 32768; mixer sub-batches of 1024), borrowing 6775 MB of the expert pool while a prompt runs (activations 5424, weight scratch 96, expert staging 86, prestage 1153: 160 experts); disk landing ring 12 experts (86 MB pinned)
RE_PREFILL_CHUNKING = re.compile(
    r"glm prefill: \S+ chunks of (\d+) tokens \([^)]*\), borrowing (\d+) MB of the expert pool[^)]*"
    r"\(activations (\d+), weight scratch (\d+), expert staging (\d+), prestage (\d+): (\d+) experts\); "
    r"disk landing ring (\d+) experts \((\d+) MB pinned\)")
# glm prefill: CUDA0 cuBLAS warm-up: sgemm 0, gemm f16 0, sync no error
RE_PREFILL_WARMUP = re.compile(r"glm prefill: \S+ cuBLAS warm-up: sgemm (\d+), gemm f16 (\d+), sync (no error|error)")
# glm prefill: CUDA0 the MMQ scratch pool has 16.0 MB of the loan (16.0 MB past the layout's need; 850 experts more lent for it)
RE_MMQ = re.compile(r"glm prefill: \S+ the MMQ scratch pool has ([\d.]+) MB of the loan \(([\d.]+) MB past[^;]*; (\d+) experts more lent")
# glm prefill: CUDA0 the chunk stage line is on (STRATA_GLM_PREFILL_STAGES=0 silences it)
RE_PREFILL_STAGE_NOTE = re.compile(r"glm prefill: \S+ the chunk stage line is on \(([^)]*)\)")
# glm prefill: CUDA0 the expert pool (0 slots a layer) can lend no prompt chunk (--prefill 16384) - token by token
RE_PREFILL_LEND_NOTE = re.compile(r"glm prefill: \S+ the expert pool \((\d+) slots a layer\) can lend no prompt chunk \(([^)]*)\)")
# glm prefill: CUDA0 80% of the experts' bytes are held pinned: the pool lends at most 85% of its slots
RE_PINNED = re.compile(r"glm prefill: \S+ ([\d.]+)% of the experts' bytes are held pinned: the pool lends at most ([\d.]+)%")
# glm stat: decode 77.48 ms/tok (12.91 tok/s) | vram hit 98.07% | ram fetches 0.17/tok | disk 6.33 reads/tok in 5.69 waits/tok, 9.88 ms/tok | promotions 6.36/tok | cpu lane 0.26/tok in 0.17 ms/tok | prompt 0.2 ms/tok | vram 9905/10012 ram 49/63
RE_STAT = re.compile(
    r"glm stat: decode ([\d.]+) ms/tok \(([\d.]+) tok/s\) \| vram hit ([\d.]+)% \| ram fetches ([\d.]+)/tok \| "
    r"disk ([\d.]+) reads/tok in ([\d.]+) waits/tok, ([\d.]+) ms/tok \| promotions ([\d.]+)/tok")
RE_STAT_TAIL = re.compile(
    r"\| cpu lane ([\d.]+)/tok in ([\d.]+) ms/tok \| prompt ([\d.]+) ms/tok \| vram (\d+)/(\d+) ram (\d+)/(\d+)")
# glm spec: 215 of 239 drafts accepted (90.0%)
RE_SPEC = re.compile(r"glm spec: (\d+) of (\d+) drafts accepted \(([\d.]+)%\)")
# glm spec: 18024 speculative positions, 90.7% accepted
RE_SPEC_POS = re.compile(r"glm spec: ([\d,]+) speculative positions, ([\d.]+)% accepted")
# glm kv: KV streaming: 99.47% of 16356615 block reads hit VRAM, 341.1 MiB read from RAM
RE_KV = re.compile(r"glm kv: KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
# glm pack: CUDA0 KV streaming: 32768 of 262144 positions of each of 11 DSA layers in VRAM (0.37 GB), the latent cache in 2.95 GB of pinned RAM
RE_PACK_KV = re.compile(
    r"glm pack: \S+ KV streaming: (\d+) of (\d+) positions of each of (\d+) (?:DSA|MLA) layers in VRAM \(([\d.]+) GB\), "
    r"the latent cache in ([\d.]+) GB of pinned RAM")
# glm pack: CUDA0 latent cache INT8 (544 bytes a position and layer)
RE_PACK_LATENT = re.compile(r"glm pack: \S+ latent cache (\w+) \((\d+) bytes a position and layer\)")
# glm fast: CUDA0 warming the expert tiers 10% (6.8 of 67.6 GB, 1 s)
RE_WARM_PCT = re.compile(r"glm fast: \S+ warming the expert tiers ([\d.]+)% \(([\d.]+) of ([\d.]+) GB, ([\d.]+) s\)")
# glm fast: CUDA0 tiers warm: 9493 experts (67.6 GB) in 8 s (8.69 GB/s)
RE_WARM_DONE = re.compile(r"glm fast: \S+ tiers warm: (\d+) experts \(([\d.]+) GB\) in ([\d.]+) s \(([\d.]+) GB/s\)")
# glm fast: CUDA0 VRAM before the expert pool: 35.72 of 119.07 GB in use - dense weights 6.86, state/KV (262144 ctx) 0.92, activations 0.06, other (context, MTP, scratch, prompt path) 27.89
RE_POOL_VRAM = re.compile(
    r"glm fast: \S+ VRAM before the expert pool: ([\d.]+) of ([\d.]+) GB in use - dense weights ([\d.]+), "
    r"state/KV \((\d+) ctx\) ([\d.]+), activations ([\d.]+), other \([^)]*\) ([\d.]+)")
# glm fast: CUDA0 unified memory: MemAvailable 96.69 GB, headroom 32.00 GB + 1024 MB reserve, experts need 80.17 GB, pool budget 63.69 GB
RE_UNIFIED = re.compile(
    r"glm fast: \S+ unified memory: MemAvailable ([\d.]+) GB, headroom ([\d.]+) GB \+ (\d+) MB reserve, "
    r"experts need ([\d.]+) GB, pool budget ([\d.]+) GB")
# glm fast: CUDA0 layers [0,45) expert pool 63.68 GB, 184-284 slots/layer (9595 total)
RE_POOL_LAYOUT = re.compile(r"glm fast: \S+ layers \[([\d,]+),([\d,]+)\) expert pool ([\d.]+) GB, (\d+)-(\d+) slots/layer \((\d+) total\)")
# glm fast: CUDA0 RAM tier 0.22 GB pinned, 32 slots
RE_RAM_TIER = re.compile(r"glm fast: \S+ RAM tier ([\d.]+) GB pinned, (\d+) slots")
# glm fast: CUDA0 CPU lane: 13 threads, an expert 0.161 ms on the CPU vs 0.127 ms over PCIe -> of 0..8 RAM-tier experts the CPU takes 0 0 1 1 2 2 2 3 3
RE_CPU_LANE = re.compile(
    r"glm fast: \S+ CPU lane: (\d+) threads, an expert ([\d.]+) ms on the CPU vs ([\d.]+) ms over PCIe -> "
    r"of (\d+)\.\.(\d+) RAM-tier experts the CPU takes ([\d ]+)$")
# glm fast: CUDA0 CPU lane timed again in decode: an expert 0.598 ms (planned with 0.162) -> RAM-tier experts to the CPU, of 0..8: 0 0 0 0 0 0 1 1 1 (PCIe share 0.95)
RE_CPU_LANE_DEC = re.compile(
    r"glm fast: \S+ CPU lane timed again in decode: an expert ([\d.]+) ms \(planned with ([\d.]+)\) -> "
    r"RAM-tier experts to the CPU, of (\d+)\.\.(\d+): ([\d ]+)\(PCIe share ([\d.]+)\)")
# glm fast: CUDA0 disk misses stream to the GPU behind their reads (8 chunks a part; MAYA_GLM_STREAM_FETCH=0 turns it off)
RE_STREAM_FETCH = re.compile(r"glm fast: \S+ disk misses stream to the GPU behind their reads \((\d+) chunks a part")
# glm fast: CUDA0 a verify window's tokens on two CPU lanes of 13 and 6 threads, unpinned
RE_VERIFY_LANES = re.compile(r"glm fast: \S+ a verify window's tokens on two CPU lanes of (\d+) and (\d+) threads, (unpinned|pinned)")
# glm fast: the expert profile /home/.../expert-profile-glm.bin ranks 13248 (layer, expert) pairs
RE_PROFILE = re.compile(r"glm fast: the expert profile (\S+) ranks (\d+) \(layer, expert\) pairs")
# glm fast: the model's clean file pages released: 89.84 GiB mapped, 89.84 GiB cached; the device's free figure 71.65 -> 77.80 GiB (on a unified-memory APU it counts them as used)
RE_CLEAN_PAGES = re.compile(
    r"glm fast: the model's clean file pages released: ([\d.]+) GiB mapped, ([\d.]+) GiB cached; "
    r"the device's free figure ([\d.]+) -> ([\d.]+) GiB")
# glm dflash: 5 layers, hidden 4096, 32 heads (8 kv), block 8, window 2048, selector top-16 rank 256, reads trunk layers 6,15,25,34,43; 717 MB on CUDA0
RE_DFLASH = re.compile(
    r"glm dflash: (\d+) layers, hidden (\d+), (\d+) heads \((\d+) kv\), block (\d+), window (\d+), "
    r"selector top-(\d+) rank (\d+), reads trunk layers ([\d,]+); ([\d.]+) MB on (\S+)")
# glm dflash: MTP off - the NextN draft block is not loaded, DFlash drafts instead
RE_DFLASH_OFF = re.compile(r"glm dflash: MTP off - the NextN draft block is not loaded, DFlash drafts instead")
# glm dflash: the head (type 14, 7 columns) 2.74 ms on the dp4a kernel, 2.75 ms on the tensor cores (medians of 6) -> dp4a
RE_DFLASH_KERNEL = re.compile(
    r"glm dflash: the head \(type (\d+), (\d+) columns\) ([\d.]+) ms on the dp4a kernel, ([\d.]+) ms on the tensor cores "
    r"\(medians of \d+\) -> (\w+)")
# glm dsa scores: CUDA0 on the original kernels
RE_DSA = re.compile(r"glm dsa scores: (\S+ on the original kernels|.*kernel.*|.*$)")
# glm_batch: CUDA0 prompt attention on the tensor cores (mma.sync, 32 heads a block)
RE_BATCH = re.compile(r"glm_batch: (\S+) prompt attention on the (tensor cores|\S+) \(([^)]*)\)")
# glm settings from the environment: MAYA_GLM_RAM_HEADROOM_GB=32, MAYA_GLM_SLOTS=1
RE_ENV = re.compile(r"glm settings from the environment: (.*)")
# glm vram: prompt 18 tokens read in 0.6 s; VRAM free CUDA0 24.2 of 119.1 GB before, 24.0 after
RE_VRAM_PROMPT = re.compile(
    r"glm vram: prompt ([\d,]+) tokens read in ([\d.]+) s; VRAM free (\S+) ([\d.]+) of ([\d.]+) GB before, ([\d.]+) after")
# glm window: a window's tokens took more scratch slots than a layer has - that window runs again token by token (said once)
RE_WINDOW_WARN = re.compile(r"glm window: (.*)")
# glm slots: saved a 26195-token conversation (its system prompt, ...) (0.48 GB in 0.3 s); 2 kept, 1.69 GB
# [maya] sessions: saved a 1738-token conversation (its system prompt, for other chats that start with it) (0.17 GB in 0.1 s); 1 kept, 0.17 GB
RE_SLOT_SAVE = re.compile(
    r"(?:glm slots|\[maya\] sessions): saved a ([\d,]+)-token conversation(?: \([^)]*\))? "
    r"\(([\d.]+) GB in ([\d.]+) s\); (\d+) kept(?:, ([\d.]+) GB)?")
# glm slots: took back a 26200-token conversation from disk (0.1 s); 226 new tokens to read
# [maya] sessions: took back a 1738-token conversation from disk (0.1 s); 226 new tokens to read
RE_SLOT_TAKE = re.compile(
    r"(?:glm slots|\[maya\] sessions): took back a ([\d,]+)-token conversation from disk \(([\d.]+) s\)"
    r"(?:; ([\d,]+) new tokens to read)?")
# glm checkpoints: 3352 of the prompt's 50511 tokens from a checkpoint
RE_CKPT = re.compile(r"glm checkpoints: ([\d,]+) of the prompt's ([\d,]+) tokens from a checkpoint")
# glm checkpoints: the cancelled read keeps 45088 of the prompt's 45123 tokens
RE_CKPT_CANCEL = re.compile(r"glm checkpoints: the cancelled read keeps ([\d,]+) of the prompt's ([\d,]+) tokens")
# [maya] thinking: 4452 of max 32000 tokens, 10.9 tok/s, 442 s   | answering: ... | writing a tool call: bash: ... | tool call complete: ...
RE_PHASE = re.compile(r"\[maya\] (thinking|answering|writing a tool call: ([^:]+)|tool call complete): "
                      r"([\d,]+) of max ([\d,]+) tokens, ([\d.]+) tok/s, ([\d,]+) s")
# [maya] thinking reached its budget (4452 tokens): closed, answering
RE_THINK_BUDGET = re.compile(r"\[maya\] thinking reached its budget \(([\d,]+) tokens\): ([^)]*\)?\S*|closed.*|)")
# [maya] done: 4930 tokens in 487 s (10.9 tok/s) (stop, cancel=False), expert cache 95.5% hit
RE_DONE = re.compile(r"\[maya\] done: ([\d,]+) tokens in ([\d,]+) s \(([\d.]+) tok/s\) \(([^,]+), cancel=(\w+)\)"
                     r"(?:, expert cache ([\d.]+)% hit)?")
# [maya] tool bash: ok, 12,345 characters in 1.2 s (truncated for the model)
RE_TOOL = re.compile(r"\[maya\] tool ([^:]+): (ok|error), ([\d,]+) characters in ([\d.]+) s(.*)")
# [maya] prompt cap: refused a 82,770-token prompt (cap 65,536; msgs=30 tools=80 last='tool':'...')
RE_REFUSED = re.compile(r"\[maya\] prompt cap: refused a ([\d,]+)-token (?:prompt|conversation) \(cap ([\d,]+); msgs=(\d+) tools=(\d+) "
                        r"last='([^']+?)['\"]:([\s\S]*)\)")
# [maya] WARNING: 2 request(s) have waited for the engine for up to 495 s ...
RE_WARN = re.compile(r"\[maya\] WARNING: (\d+) request\(s\) have waited for the engine for up to ([\d,]+) s")
# [maya] waiting for room: 6069 MiB free, 7489 MiB needed (the borrow 4417 MiB + the 3072 MiB floor)
RE_ROOM = re.compile(r"\[maya\] waiting for room: ([\d,]+) MiB free, ([\d,]+) MiB needed "
                     r"\(the borrow ([\d,]+) MiB \+ the ([\d,]+) MiB floor\)")
# [maya] this request waited 5 s for the engine (0 request(s) behind it)
RE_WAITED = re.compile(r"\[maya\] this request waited ([\d.]+) s for the engine \((\d+) request\(s\) behind it\)")
# [maya] raw: '...'  (the full answer; keep its length only)
RE_RAW = re.compile(r"\[maya\] raw: (?:'|\")(.*?)(?:'|\")\s*$", re.S)
# [maya] model management: on (/models/load, /models/unload, /models/sse)
RE_MODEL_MGMT = re.compile(r"\[maya\] model management: on \((.*)\)")
# [maya] thinking level when a request names none: high
RE_THINK_LEVEL = re.compile(r"\[maya\] thinking level when a request names none: (\w+)")
# [maya] thinking budget: 32768 tokens (then the answer)
RE_THINK_BUDGET_CFG = re.compile(r"\[maya\] thinking budget: ([\d,]+) tokens \(([^)]*)\)")
# [maya] sampling defaults from the config: temperature=1.0, top_p=0.95
RE_SAMPLING = re.compile(r"\[maya\] sampling defaults from the config: (.*)")
# [maya] this server's lines go to /home/.../maya-maya-s-v2-iq2_xxs-256k.log too
RE_SERVER_LINES = re.compile(r"\[maya\] this server's lines go to (\S+)")
# [maya] an engine from an earlier start still holds the GPU (pids [299187]); ending it
RE_ENGINE_PREEMPT = re.compile(r"\[maya\] an engine from an earlier start still holds the GPU \(pids \[([^\]]*)\]\)")
# loading the model (the first start takes a minute or two) ...
RE_LOADING = re.compile(r"loading the model \(the first start takes a minute or two\)")
#        open http://127.0.0.1:8009/ in a browser to chat; close this window to stop the model
RE_OPEN_BROWSER = re.compile(r"open (http://[\d.]+:\d+)/ in a browser")
# Traceback (most recent call last): / File "...", line N
RE_TRACEBACK = re.compile(r"(Traceback \(most recent call last\):|\s+File \"(.+)\", line (\d+))")
# [maya] the GPU readings' NVML reader stopped (its pipe broke (...)); starting it again
RE_NVML = re.compile(r"\[maya\] the GPU readings' NVML reader stopped")
# maya serve: CUDA0: 18358 MiB of VRAM free with everything loaded
RE_SERVE_FREE = re.compile(r"maya serve: (\S+): ([\d,]+) MiB of VRAM free with everything loaded")
# maya serve: stall report (engine 0.1.27): stage "the request -1" for 59 s; 0 layers served since the last finished step
RE_STALL = re.compile(r"maya serve: stall report \(engine (\S+)\): stage \"([^\"]*)\" for ([\d.]+) s; (\d+) layers served")
# maya serve: no progress for 180 s during a request (the request -1) - stopping the engine ...
RE_NO_PROGRESS = re.compile(r"maya serve: no progress for (\d+) s during a request \(the request ([\d-]+)\)")
# maya serve: suffix drafts: 26 windows, 43 of 59 drafts accepted
RE_SUFFIX_MAYA = re.compile(r"maya serve: suffix drafts: (\d+) windows, ([\d,]+) of ([\d,]+) drafts accepted")
# maya generate: glm5-next serve (M3.1): GEN <max_new> [keys] <ids>, GENI <max_new> [keys] <embeddings> <ids>; QUIT to end
RE_GEN_BANNER = re.compile(r"maya generate: (\S+) serve \(M([\d.]+)\): (GEN.*)")
# maya generate: glm fast: CUDA0 has no room for the expert pool ...
RE_GEN_ROOM = re.compile(r"maya generate: (.*has no room.*)")
# memory: 123456 MiB resident, 0 MiB in swap, 23456 MiB RAM available; ...
RE_MEM = re.compile(r"memory: ([\d,]+) MiB resident, ([\d,]+) MiB in swap, ([\d,]+) MiB RAM available")
# from other devices: http://10.0.0.1:8009/   (API: http://10.0.0.1:8009/v1)
RE_OTHER_DEV = re.compile(r"from other devices: (\S+)/\s+\(API: (\S+)\)")

# ============================================================ strata =======
# (secondary dialect; kept from the A5000 deployment)
RE_STRATA_PROMPT = re.compile(
    r"strata serve: prompt ([\d,]+) tokens = ([\d,]+) reused \+ ([\d,]+) read in "
    r"([\d,.]+) ms \(([\d.]+) tok/s\), ([\d,]+) generated in ([\d,.]+) ms \(([\d.]+) tok/s\), "
    r"drafts accepted ([\d,]+) of ([\d,]+), (\d+) checkpoints")
RE_STRATA_EXPERT = re.compile(
    r"strata serve: decode expert cache hit rate: ([\d.]+)% \(([\d,]+) hits / ([\d,]+) lookups\)")
RE_STRATA_KV = re.compile(
    r"strata serve: KV streaming: ([\d.]+)% of ([\d,]+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
RE_STRATA_SUFFIX = re.compile(
    r"strata serve: suffix drafts: (\d+) windows, ([\d,]+) of ([\d,]+) drafts accepted")
RE_STRATA_PARK = re.compile(
    r"strata serve: conversation cache: parked ([\d,]+) tokens in ([\d.]+) ms; parked=(\d+) bytes=(\d+)")
RE_STRATA_RESTORE = re.compile(
    r"strata serve: conversation cache: restored ([\d,]+) tokens \((live|checkpoint)\) in ([\d.]+) ms")
RE_STRATA_BORROW = re.compile(r"strata serve: conversation cache: borrowed ([\d,]+) tokens")
RE_STRATA_VRAM_WARN = re.compile(r"strata serve: WARNING: only ([\d,]+) MiB of VRAM free .*")
RE_STRATA_ENGINE = re.compile(r"\[strata\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d engine (started|stopped|exited)")
RE_STRATA_MTP = re.compile(
    r"strata mtp: draft layer loaded, ([\d,]+) MiB of VRAM \(experts ([\d,]+), dense ([\d,]+)\), "
    r"files read in ([\d.]+) s \(([\d,]+) MiB/s\)")
RE_STRATA_PREFILL_SLOTS = re.compile(
    r"strata generate: pre-filled ([\d,]+) of ([\d,]+) slots from the profile in ([\d.]+) s")
RE_STRATA_WORKERS = re.compile(r"strata generate: (\d+) expert-pool workers")
RE_STRATA_SESSION = re.compile(r"strata generate: session is up \(engine (\S+)\)")
RE_STRATA_VERIFY = re.compile(r"strata verify: window up to (\d+) tokens, ([\d.]+) MiB of device buffers")


def parse_line(line: str):
    """One log line -> dict, list of dicts, or None."""
    line = line.rstrip("\n")
    if not line.strip():
        return None

    # ---- strata completion-summary dialect (secondary) --------------------
    m = RE_STRATA_PROMPT.search(line)
    if m:
        prompt, reused, read, read_ms, read_ts, gen, gen_ms, gen_ts = map(_n, m.groups()[:8])
        acc, drafts, ckpts = (int(_n(v)) for v in m.groups()[8:])
        out = [{"type": "req", "api": "strata", "msgs": None, "tools": None,
                "max_new": None, "thinking": None, "stream": None,
                "prompt_tokens": prompt, "reused": reused, "read": read,
                "checkpoints": ckpts, "timing": "completion_summary", "last_role": "-",
                "last_preview": f"{int(reused):,} reused + {int(read):,} read"}]
        if reused:
            out.append({"type": "reuse", "reused": reused, "prompt_tokens": prompt})
        out.append({"type": "prefill_sum", "tokens": read, "ms": read_ms,
                    "ms_tok": round(read_ms / read, 3) if read else None,
                    "tok_s": read_ts if read else None, "chunks": None,
                    "chunk_tokens": None, "timing": "completion_summary"})
        if drafts:
            out.append({"type": "spec", "source": "mtp", "accepted": acc, "drafts": drafts,
                        "pct": round(100 * acc / drafts, 1)})
        out.append({"type": "done", "tokens": gen, "seconds": gen_ms / 1000,
                    "tok_s": gen_ts, "finish": None, "cancel": None,
                    "cache_pct": None, "timing": "completion_summary"})
        return out
    m = RE_STRATA_EXPERT.search(line)
    if m:
        return {"type": "expert_cache", "hit_pct": _n(m.group(1)),
                "hits": _n(m.group(2)), "lookups": _n(m.group(3))}
    m = RE_STRATA_KV.search(line)
    if m:
        return {"type": "kv", "hit_pct": _n(m.group(1)), "blocks": int(_n(m.group(2))),
                "ram_mib": _n(m.group(3))}
    m = RE_STRATA_SUFFIX.search(line)
    if m:
        windows, acc, drafts = (int(_n(v)) for v in m.groups())
        return {"type": "spec", "source": "suffix", "accepted": acc, "drafts": drafts,
                "pct": round(100 * acc / drafts, 1) if drafts else None, "windows": windows}
    m = RE_STRATA_PARK.search(line)
    if m:
        return {"type": "slot_save", "tokens": _n(m.group(1)),
                "cache_gb": round(int(m.group(4)) / 1e9, 2),
                "s": _n(m.group(2)) / 1000, "kept": int(m.group(3))}
    m = RE_STRATA_RESTORE.search(line)
    if m:
        return {"type": "cache_restore", "tokens": _n(m.group(1)),
                "ms": _n(m.group(3)), "restore": m.group(2)}
    m = RE_STRATA_BORROW.search(line)
    if m:
        return {"type": "borrow", "tokens": _n(m.group(1))}
    m = RE_STRATA_VRAM_WARN.search(line)
    if m:
        return {"type": "vram_warn", "mib": _n(m.group(1))}
    m = RE_STRATA_ENGINE.search(line)
    if m:
        return {"type": "engine_start" if m.group(1) == "started" else "engine_exit", "text": line[:160]}
    m = RE_STRATA_MTP.search(line)
    if m:
        return {"type": "mtp_load", "vram_mib": _n(m.group(1)), "experts_mib": _n(m.group(2)),
                "dense_mib": _n(m.group(3)), "read_s": _n(m.group(4)), "mib_s": _n(m.group(5))}
    m = RE_STRATA_PREFILL_SLOTS.search(line)
    if m:
        return {"type": "expert_prefill", "filled": _n(m.group(1)), "slots": _n(m.group(2)),
                "s": _n(m.group(3))}
    m = RE_STRATA_WORKERS.search(line)
    if m:
        return {"type": "pool_workers", "workers": int(m.group(1))}
    m = RE_STRATA_SESSION.search(line)
    if m:
        return {"type": "session_up", "engine": m.group(1)}
    m = RE_STRATA_VERIFY.search(line)
    if m:
        return {"type": "verify_setup", "window": int(m.group(1)), "mib": _n(m.group(2))}
    if line.startswith("strata-") and ": ready instance " in line:
        return {"type": "engine_ready", "text": line[:160]}
    if line.startswith("strata-") and line.endswith(": stopped"):
        return {"type": "engine_exit", "text": line[:160]}

    # ---- maya / glm dialect (primary) -------------------------------------
    m = RE_REQ.search(line)
    if m:
        return {"type": "req", "api": m.group(1), "msgs": int(m.group(2)), "tools": int(m.group(3)),
                "max_new": _n(m.group(5)), "thinking": m.group(6) == "True", "stream": m.group(7) == "True",
                "prompt_tokens": _n(m.group(8)), "last_role": m.group(9),
                "last_preview": m.group(10)[:70]}
    m = RE_READ.search(line)
    if m:
        return {"type": "read", "done": _n(m.group(1)),
                "total": _n(m.group(2)) or _n(m.group(1)), "elapsed": _n(m.group(3))}
    m = RE_READ2.search(line)
    if m:
        return {"type": "read", "done": _n(m.group(1)), "total": _n(m.group(2)),
                "tok_s": _n(m.group(3)), "elapsed": _n(m.group(4))}
    m = RE_CHUNK.search(line)
    if m:
        ev = {"type": "prefill_chunk", "chunk": int(m.group(1)), "chunks": int(m.group(2)),
                "tokens": _n(m.group(3)), "wall_ms": _n(m.group(4)), "copy_ms": _n(m.group(5)),
                "copies": int(m.group(6)), "copy_gb_s": _n(m.group(7)), "pcie_gb": _n(m.group(8)),
                "pcie_ram": _n(m.group(9)), "pcie_disk": _n(m.group(10)),
                "vram_gb": _n(m.group(11)), "staged": int(m.group(12)),
                "resident": int(m.group(13))}
        if m.group(14):
            ev["resident_rows"] = int(_n(m.group(14)))
        if m.group(15):
            ev["prestaged"] = int(m.group(15))
        if m.group(16):
            ev["cpu_experts"] = int(m.group(16))
        # extras: ring wait / gaps / mixer windows / lead / tail / layer lane
        x = re.search(r"ring wait (\d+), gaps (\d+), mixer windows (\d+) \| lead (\d+), tail (\d+)", line)
        if x:
            ev.update(ring_wait=int(x.group(1)), gaps=int(x.group(2)),
                      mixer_windows=int(x.group(3)), lead=int(x.group(4)), tail=int(x.group(5)))
        x = re.search(r"layer lane ([\d.]+)/([\d.]+) ms max/min", line)
        if x:
            ev.update(layer_lane_max=float(x.group(1)), layer_lane_min=float(x.group(2)))
        return ev
    m = RE_PREFILL_SUM.search(line)
    if m:
        return {"type": "prefill_sum", "tokens": _n(m.group(1)), "ms": _n(m.group(2)),
                "ms_tok": _n(m.group(3)), "tok_s": _n(m.group(4)), "chunks": int(m.group(5)),
                "chunk_tokens": _n(m.group(6))}
    m = RE_PREFILL_CHUNKING.search(line)
    if m:
        return {"type": "prefill_chunking", "chunk_tokens": int(m.group(1)),
                "borrowed_mb": int(m.group(2)), "act_mb": int(m.group(3)),
                "scratch_mb": int(m.group(4)), "staging_mb": int(m.group(5)),
                "prestage_mb": int(m.group(6)), "prestage_experts": int(m.group(7)),
                "ring_experts": int(m.group(8)), "ring_mb": int(m.group(9))}
    m = RE_PREFILL_WARMUP.search(line)
    if m:
        return {"type": "prefill_warmup", "sgemm": int(m.group(1)), "gemm_f16": int(m.group(2)),
                "sync": m.group(3)}
    m = RE_MMQ.search(line)
    if m:
        return {"type": "mmq_scratch", "loan_mb": _n(m.group(1)), "past_mb": _n(m.group(2)),
                "experts_lent": int(m.group(3))}
    m = RE_PREFILL_STAGE_NOTE.search(line)
    if m:
        return {"type": "prefill_stage_note", "env": m.group(1)}
    m = RE_PREFILL_LEND_NOTE.search(line)
    if m:
        return {"type": "prefill_lend_note", "slots_per_layer": int(m.group(1)), "hint": m.group(2)}
    m = RE_PINNED.search(line)
    if m:
        return {"type": "experts_pinned", "pct": _n(m.group(1)), "lend_pct": _n(m.group(2))}
    m = RE_STAT.search(line)
    if m:
        ev = {"type": "stat", "ms_tok": _n(m.group(1)), "tok_s": _n(m.group(2)),
                "vram_hit": _n(m.group(3)), "ram_fetch": _n(m.group(4)),
                "disk_reads_tok": _n(m.group(5)), "disk_waits_tok": _n(m.group(6)),
                "disk_ms_tok": _n(m.group(7)), "promotions_tok": _n(m.group(8))}
        x = RE_STAT_TAIL.search(line)
        if x:
            ev.update(cpu_lane_rate=_n(x.group(1)), cpu_lane_ms=_n(x.group(2)),
                      prompt_ms_tok=_n(x.group(3)), vram_used=int(x.group(4)),
                      vram_slots=int(x.group(5)), ram_used=int(x.group(6)),
                      ram_slots=int(x.group(7)))
        return ev
    m = RE_SPEC.search(line)
    if m:
        return {"type": "spec", "source": "dflash", "accepted": int(m.group(1)),
                "drafts": int(m.group(2)), "pct": _n(m.group(3))}
    m = RE_SPEC_POS.search(line)
    if m:
        return {"type": "spec_positions", "positions": _n(m.group(1)), "pct": _n(m.group(2))}
    m = RE_KV.search(line)
    if m:
        return {"type": "kv", "hit_pct": _n(m.group(1)), "blocks": int(_n(m.group(2))),
                "ram_mib": _n(m.group(3))}
    m = RE_PACK_KV.search(line)
    if m:
        return {"type": "pack_kv", "positions_vram": int(m.group(1)),
                "positions_total": int(m.group(2)), "layers": int(m.group(3)),
                "vram_gb": _n(m.group(4)), "latent_ram_gb": _n(m.group(5))}
    m = RE_PACK_LATENT.search(line)
    if m:
        return {"type": "pack_latent", "dtype": m.group(1), "bytes_per_pos_layer": int(m.group(2))}
    m = RE_WARM_PCT.search(line)
    if m:
        return {"type": "tier_warm", "pct": _n(m.group(1)), "gb_done": _n(m.group(2)),
                "gb_total": _n(m.group(3)), "s": _n(m.group(4))}
    m = RE_WARM_DONE.search(line)
    if m:
        return {"type": "tiers_warm", "experts": int(m.group(1)), "gb": _n(m.group(2)),
                "s": _n(m.group(3)), "gb_s": _n(m.group(4))}
    m = RE_POOL_VRAM.search(line)
    if m:
        return {"type": "pool_vram", "used_gb": _n(m.group(1)), "total_gb": _n(m.group(2)),
                "dense_gb": _n(m.group(3)), "ctx": int(_n(m.group(4))),
                "state_kv_gb": _n(m.group(5)), "act_gb": _n(m.group(6)),
                "other_gb": _n(m.group(7))}
    m = RE_UNIFIED.search(line)
    if m:
        return {"type": "unified_mem", "mem_available_gb": _n(m.group(1)),
                "headroom_gb": _n(m.group(2)), "reserve_mb": int(m.group(3)),
                "experts_need_gb": _n(m.group(4)), "pool_budget_gb": _n(m.group(5))}
    m = RE_POOL_LAYOUT.search(line)
    if m:
        return {"type": "pool_layout", "layer_lo": int(m.group(1)), "layer_hi": int(m.group(2)),
                "pool_gb": _n(m.group(3)), "slots_min": int(m.group(4)),
                "slots_max": int(m.group(5)), "slots_total": int(m.group(6))}
    m = RE_RAM_TIER.search(line)
    if m:
        return {"type": "ram_tier", "gb": _n(m.group(1)), "slots": int(m.group(2))}
    m = RE_CPU_LANE.search(line)
    if m:
        return {"type": "cpu_lane", "threads": int(m.group(1)),
                "expert_ms_cpu": _n(m.group(2)), "expert_ms_pcie": _n(m.group(3)),
                "range_lo": int(m.group(4)), "range_hi": int(m.group(5)),
                "takes": m.group(6).split()}
    m = RE_CPU_LANE_DEC.search(line)
    if m:
        return {"type": "cpu_lane_decode", "expert_ms": _n(m.group(1)),
                "planned_ms": _n(m.group(2)), "range_lo": int(m.group(3)),
                "range_hi": int(m.group(4)), "takes": m.group(5).split(),
                "pcie_share": _n(m.group(6))}
    m = RE_STREAM_FETCH.search(line)
    if m:
        return {"type": "stream_fetch", "chunks_per_part": int(m.group(1))}
    m = RE_VERIFY_LANES.search(line)
    if m:
        return {"type": "verify_lanes", "lane_a": int(m.group(1)), "lane_b": int(m.group(2)),
                "pinning": m.group(3)}
    m = RE_PROFILE.search(line)
    if m:
        return {"type": "profile_loaded", "path": m.group(1), "pairs": int(_n(m.group(2)))}
    m = RE_CLEAN_PAGES.search(line)
    if m:
        return {"type": "clean_pages", "gib_mapped": _n(m.group(1)), "gib_cached": _n(m.group(2)),
                "free_before_gb": _n(m.group(3)), "free_after_gb": _n(m.group(4))}
    m = RE_DFLASH.search(line)
    if m:
        return {"type": "dflash_setup", "layers": int(m.group(1)), "hidden": int(m.group(2)),
                "heads": int(m.group(3)), "kv_heads": int(m.group(4)), "block": int(m.group(5)),
                "window": int(m.group(6)), "selector_top": int(m.group(7)),
                "rank": int(m.group(8)),
                "trunk_layers": [int(v) for v in m.group(9).split(",")],
                "mb": _n(m.group(10)), "dev": m.group(11)}
    m = RE_DFLASH_OFF.search(line)
    if m:
        return {"type": "dflash_mtp_off", "text": line[:120]}
    m = RE_DFLASH_KERNEL.search(line)
    if m:
        return {"type": "dflash_kernel", "head_type": int(m.group(1)), "columns": int(m.group(2)),
                "dp4a_ms": _n(m.group(3)), "tc_ms": _n(m.group(4)), "chosen": m.group(5)}
    m = RE_DSA.search(line)
    if line.startswith("glm dsa scores:"):
        return {"type": "dsa_kernels", "text": line[:120]}
    m = RE_BATCH.search(line)
    if line.startswith("glm_batch:"):
        return {"type": "batch_kernels", "dev": m.group(1), "device": m.group(2),
                "note": m.group(3)}
    m = RE_ENV.search(line)
    if m:
        return {"type": "env_settings", "raw": m.group(1)[:160]}
    m = RE_VRAM_PROMPT.search(line)
    if m:
        return {"type": "vram_prompt", "tokens": _n(m.group(1)), "s": _n(m.group(2)),
                "dev": m.group(3), "free_before_gb": _n(m.group(4)),
                "total_gb": _n(m.group(5)), "free_after_gb": _n(m.group(6))}
    m = RE_WINDOW_WARN.search(line)
    if line.startswith("glm window:"):
        return {"type": "window_warn", "text": line[:120]}
    m = RE_SLOT_SAVE.search(line)
    if m:
        return {"type": "slot_save", "tokens": _n(m.group(1)), "gb": _n(m.group(2)),
                "s": _n(m.group(3)), "kept": int(m.group(4)),
                "kept_gb": _n(m.group(5))}
    m = RE_SLOT_TAKE.search(line)
    if m:
        return {"type": "slot_take", "tokens": _n(m.group(1)), "s": _n(m.group(2)),
                "new_tokens": _n(m.group(3))}
    m = RE_CKPT_CANCEL.search(line)
    if m:
        return {"type": "ckpt", "reused": _n(m.group(1)), "prompt_tokens": _n(m.group(2)), "cancelled": True}
    m = RE_CKPT.search(line)
    if m:
        return {"type": "ckpt", "reused": _n(m.group(1)), "prompt_tokens": _n(m.group(2))}
    m = RE_PHASE.search(line)
    if m:
        return {"type": "phase", "phase": (m.group(2) or m.group(1)).strip(),
                "tokens": _n(m.group(3)), "max": _n(m.group(4)),
                "tok_s": _n(m.group(5)), "elapsed": _n(m.group(6))}
    m = RE_THINK_BUDGET.search(line)
    if line.startswith("[maya] thinking reached its budget"):
        return {"type": "think_budget", "tokens": _n(m.group(1)), "outcome": m.group(2)}
    m = RE_DONE.search(line)
    if m:
        return {"type": "done", "tokens": _n(m.group(1)), "seconds": _n(m.group(2)),
                "tok_s": _n(m.group(3)), "finish": m.group(4), "cancel": m.group(5) == "True",
                "cache_pct": _n(m.group(6))}
    m = RE_TOOL.search(line)
    if m:
        return {"type": "tool_result", "name": m.group(1), "ok": m.group(2) == "ok",
                "chars": _n(m.group(3)), "s": _n(m.group(4)), "truncated": "truncated" in m.group(5)}
    m = RE_REFUSED.search(line)
    if m:
        return {"type": "refused", "tokens": _n(m.group(1)), "cap": _n(m.group(2)),
                "msgs": int(m.group(3)), "tools": int(m.group(4)),
                "last_role": m.group(5), "last_preview": m.group(6)[:70]}
    m = RE_WARN.search(line)
    if m:
        return {"type": "warning", "waiters": int(m.group(1)), "waited_s": _n(m.group(2))}
    m = RE_ROOM.search(line)
    if m:
        return {"type": "room_wait", "free_mib": _n(m.group(1)), "needed_mib": _n(m.group(2)),
                "borrow_mib": _n(m.group(3)), "floor_mib": _n(m.group(4))}
    m = RE_WAITED.search(line)
    if m:
        return {"type": "waited", "s": _n(m.group(1)), "behind": int(m.group(2))}
    m = RE_MODEL_MGMT.search(line)
    if m:
        return {"type": "model_mgmt", "endpoints": m.group(1)}
    m = RE_THINK_LEVEL.search(line)
    if m:
        return {"type": "thinking_cfg", "level": m.group(1)}
    m = RE_THINK_BUDGET_CFG.search(line)
    if m:
        return {"type": "thinking_cfg", "budget_tokens": _n(m.group(1)), "then": m.group(2)}
    m = RE_SAMPLING.search(line)
    if m:
        return {"type": "sampling_cfg", "raw": m.group(1)[:120]}
    m = RE_SERVER_LINES.search(line)
    if m:
        return {"type": "log_note", "text": f"server lines go to {m.group(1)}"}
    m = RE_ENGINE_PREEMPT.search(line)
    if m:
        return {"type": "engine_preempt", "pids": m.group(1)}
    m = RE_LOADING.search(line)
    if m:
        return {"type": "engine_loading", "text": line[:80]}
    m = RE_OPEN_BROWSER.search(line)
    if m:
        return {"type": "endpoints", "http": m.group(1), "api": None}
    m = RE_TRACEBACK.match(line)
    if m:
        return {"type": "crash_traceback", "file": m.group(2), "line_no": int(m.group(3)) if m.group(3) else None}
    m = RE_NVML.search(line)
    if m:
        return {"type": "nvml_restart", "text": line[:120]}
    m = RE_SERVE_FREE.search(line)
    if m:
        return {"type": "serve_free", "dev": m.group(1), "mib_free": _n(m.group(2))}
    m = RE_STALL.search(line)
    if m:
        return {"type": "stall_report", "engine": m.group(1), "stage": m.group(2),
                "s": _n(m.group(3)), "layers_served": int(m.group(4))}
    m = RE_NO_PROGRESS.search(line)
    if m:
        return {"type": "no_progress", "s": int(m.group(1)), "request": m.group(2)}
    m = RE_SUFFIX_MAYA.search(line)
    if m:
        windows, acc, drafts = (int(_n(v)) for v in m.groups())
        return {"type": "spec", "source": "suffix", "accepted": acc, "drafts": drafts,
                "pct": round(100 * acc / drafts, 1) if drafts else None, "windows": windows}
    m = RE_GEN_BANNER.search(line)
    if m:
        return {"type": "generate_banner", "model": m.group(1), "maya_version": m.group(2),
                "banner": m.group(3)[:120]}
    m = RE_GEN_ROOM.search(line)
    if m:
        return {"type": "generate_room", "text": line[:140]}
    m = RE_MEM.search(line)
    if m:
        return {"type": "mem", "resident_mib": _n(m.group(1)), "swap_mib": _n(m.group(2)),
                "ram_avail_mib": _n(m.group(3))}
    m = RE_OTHER_DEV.search(line)
    if m:
        return {"type": "endpoints", "http": m.group(1), "api": m.group(2)}
    if RE_RAW.match(line):
        return {"type": "answer", "chars": len(line)}
    for tag, typ in (("starting the engine", "engine_start"), ("starting it again", "engine_start"),
                     ("the engine had stopped", "engine_exit"), ("ready: http", "engine_ready"),
                     ("stopping (", "engine_stopping"), ("still starting", "engine_warming"),
                     ("the engine reported an error", "engine_error"),
                     ("the engine is running again", "engine_resume"),
                     ("the engine stopped unexpectedly", "engine_crash"),
                     ("the engine did not end within", "engine_exit_timeout"),
                     ("did not say the request had ended", "engine_exit_timeout"),
                     ("stopping:", "engine_stopping"),
                     ("skipping the lines of the request", "skip_note"),
                     ("the skipped request's end reached", "skip_end"),
                     ("out of memory: the device has", "gate_refusal")):
        if tag in line:
            return {"type": typ, "text": line[:160]}
    return {"type": "log", "text": line[:160]}
