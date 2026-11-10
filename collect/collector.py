#!/usr/bin/env python3
"""
collector.py — maya telemetry bridge for the WebGL weight atlas.

The modified-maya engine exposes a single rich JSON snapshot at
    GET http://127.0.0.1:8009/metrics        (Authorization: Bearer sk-local)
(engine phase, tok/s, tier counters + 120-sample history arrays, gpu_map,
requests, totals, storage/savings) and writes a human-readable log to
    ~/projects/maya/maya-maya-s-v2-iq2_xxs-256k.log
which contains structured lines we parse into typed events (see
log_parser.py: DFlash accept ratios, expert-tier warm, KV/pack streaming,
pool layout, CPU lane, slot save/restore, room waits, engine lifecycle,
DFlash2 setup/kernel choice, thinking budget, stall/no-progress reports).

This process merges both sources into live/telemetry.json every `--interval`
seconds, and can *serve* the web/ directory plus the merged snapshot on
`--serve-port` so the visualiser needs no separate web server:

    GET /                    -> web/index.html
    GET /api/telemetry       -> live/telemetry.json content
    GET /api/atlas           -> live/atlas_manifest.json content
    GET <anything else>      -> static file under web/

Safety posture: engine metrics/log are read-only; nothing is ever POSTed to
the engine. `--once` writes a single snapshot and exits (cron-friendly).
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading, time, urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_parser import parse_line  # noqa: E402

HOME = os.path.expanduser("~")
DEF_METRICS = "http://127.0.0.1:8009/metrics"
DEF_TOKEN = os.environ.get("MAYA_METRICS_TOKEN", "sk-local")
DEF_LOG = f"{HOME}/projects/maya/maya-maya-s-v2-iq2_xxs-256k.log"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_OUT = os.path.join(ROOT, "live", "telemetry.json")
DEF_WEB = os.path.join(ROOT, "web")


def fetch_json(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {DEF_TOKEN}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class LogTailer:
    """Follows the engine log like `tail -F`, yielding parsed events.

    Survives rotation: on ENOENT/shrink, reopens from 0. Events are kept in
    a ring so the JSON snapshot carries the most recent `ring` entries.
    """
    def __init__(self, path: str, ring: int = 60, from_start: bool = False):
        self.path, self.ring = path, ring
        self.events: deque = deque(maxlen=ring)
        self.counters: dict = {}
        self.from_start = from_start
        self.f = None
        self.pos = 0
        self.open()

    def open(self):
        if self.f:
            try:
                self.f.close()
            except Exception:
                pass
        try:
            self.f = open(self.path, "r", errors="replace")
            if not self.from_start:
                self.f.seek(0, os.SEEK_END)
        except FileNotFoundError:
            self.f = None

    def pump(self):
        """Read whatever is available; return list of new event dicts."""
        new = []
        if self.f is None:
            self.open()
            return new
        while True:
            line = self.f.readline()
            if not line:
                try:
                    if os.path.getsize(self.path) < self.f.tell():
                        self.open()  # rotated
                except OSError:
                    self.open()
                break
            evs = parse_line(line.rstrip("\n"))
            if evs:
                if isinstance(evs, dict):
                    evs = [evs]
                for ev in evs:
                    ev.setdefault("t", time.strftime("%H:%M:%S"))
                    self.events.append(ev)
                    self.counters[ev["type"]] = self.counters.get(ev["type"], 0) + 1
                    new.append(ev)
        return new


def compose(metrics, tailer, started, hist=None):
    """Merge engine metrics + parsed log state into the telemetry snapshot.

    `hist` is a persistent dict of deques kept across polls by the caller;
    currently one series: kv.seq_tokens sampled every poll (KV growth curve).
    """
    if hist is None:
        hist = {"kv": deque(maxlen=180)}
    out = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collector_uptime_s": round(time.time() - started, 1),
        "events": list(tailer.events)[-40:],
        "log_counters": dict(tailer.counters),
        "sources": {
            "metrics": {"url": None, "ok": metrics is not None},
            "log": {"path": tailer.path},
        },
    }
    if metrics:
        for k in ("engine", "live", "requests", "requests_kept", "totals", "tiers",
                  "gpu_map", "storage", "savings", "hardware", "hardware_static",
                  "history", "loop_guard", "reload", "time"):
            if k in metrics:
                out[k] = metrics[k]
        eng = metrics.get("engine") or {}
        live = metrics.get("live") or {}
        now = (metrics.get("tiers") or {}).get("now") or {}
        reqs = metrics.get("requests") or []
        tot = metrics.get("totals") or {}

        # speculation acceptance: maya reports running counters per tier sample
        acc, off = now.get("draft_acc"), now.get("draft_off")
        if not off and reqs:
            acc = reqs[0].get("drafts_accepted") or 0
            off = reqs[0].get("drafts_offered") or 0
        out["derived"] = {
            "mtp_accept_rate": (acc / off) if off else None,
            "prompt_hit_rate": (tot.get("reused") / tot["prompt_tokens"])
                               if tot.get("prompt_tokens") else None,
            "speculation": eng.get("speculation") or [],   # ["DFlash2"]
        }

        # ---- (a) expert residency: every step touches 43 MoE layers x
        #          top-8 experts (main model layers 3..44 + NextN draft
        #          blk.45) ----------------------------------------------------
        exp = {"per_token_structural": 43 * 8,
               "slots_total": eng.get("experts"),          # 12,096
               "slots_vram": eng.get("vram_slots"),        # e.g. 10,012
               "slots_ram": eng.get("ram_slots"),          # e.g. 63
               "slot_pool_mib": eng.get("expert_cache_mib")}
        if now.get("vram_used") is not None:
            exp.update(resident_vram=now.get("vram_used"),
                       resident_ram=now.get("ram_used"),
                       vram_hit=now.get("vram_hit"),
                       ram_fetch=now.get("ram_fetch"),
                       disk_read=now.get("disk"),
                       promotions=now.get("promo"))
        # last request's expert-cache hit from the log ([maya] done: ..., expert cache 98.1% hit)
        for ev in reversed(list(tailer.events)):
            if ev.get("type") == "done" and ev.get("cache_pct") is not None:
                exp["last_request_hit_pct"] = ev["cache_pct"]
                break
        out["derived"]["experts"] = exp

        # ---- (c) KV cache occupancy -------------------------------------
        seq = None
        if live.get("prompt_total") is not None:
            seq = int(live["prompt_total"]) + int(live.get("generated") or 0)
        elif reqs:
            r0 = reqs[0]
            seq = int(r0.get("prompt_tokens") or 0) + int(r0.get("output_tokens") or 0)
        if seq is not None:
            hist["kv"].append(seq)
        out["kv"] = {
            "dtype": eng.get("kv"),                        # "int8"
            "capacity_tokens": eng.get("kv_ctx"),          # 262144 served
            "trained_context": eng.get("trained_context"),  # 1,048,576
            "resident_tokens": eng.get("kv_resident"),     # VRAM-resident prefix
            "seq_tokens": seq,
            "series": list(hist["kv"]),
        }

        # ---- (d) GPU map (NVIDIA GB10 + drafter + pool, NVML-measured) ---
        gm = metrics.get("gpu_map") or []
        if gm:
            g0 = gm[0]
            parts = g0.get("parts") or {}
            out["gpu"] = {
                "name": g0.get("name"),                    # "NVIDIA GB10"
                "total_gb": round(g0.get("total", 0) / 1e9, 2),
                "free_gb": round(g0.get("free", 0) / 1e9, 2),
                "pool_gb": round(parts.get("pool", 0) / 1e9, 2),
                "drafter_gb": round((g0.get("drafter") or parts.get("drafter", 0)) / 1e9, 2),
                "borrow_gb": round((g0.get("borrow") or 0) / 1e9, 2),
                "kv_ram_gb": round((g0.get("kv_ram") or 0) / 1e9, 2),
                "layers": g0.get("layers"),                # [0, 45)
                "slots": g0.get("slots"),
                "measured": g0.get("measured"),           # "nvml"
            }
    return out


class Serve:
    """Serves web/ statically; /api/telemetry and /api/atlas read live files."""
    def __init__(self, web_dir, live_dir, port):
        self.web, self.live, self.port = web_dir, live_dir, port

    def handler(self):
        web, live = self.web, self.live
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass
            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/", ""):
                    p = "/index.html"
                if p == "/api/telemetry":
                    try:
                        return self._send(open(os.path.join(live, "telemetry.json"), "rb").read(), "application/json")
                    except FileNotFoundError:
                        return self._send(b'{"state":"waiting"}', "application/json")
                if p == "/api/atlas":
                    try:
                        return self._send(open(os.path.join(live, "atlas_manifest.json"), "rb").read(), "application/json")
                    except FileNotFoundError:
                        return self._send(b'{"error":"run tools/build_atlas.py"}', "application/json")
                fp = os.path.normpath(os.path.join(web, p.lstrip("/")))
                if not fp.startswith(web) or not os.path.isfile(fp):
                    self.send_error(404)
                    return
                ctype = ("application/javascript" if fp.endswith(".js")
                        else "application/json" if fp.endswith(".json")
                        else "text/html" if fp.endswith(".html") else "text/plain")
                self._send(open(fp, "rb").read(), ctype)
        return H


def _resolve_bind(host):
    """'tailscale'/'ts' -> tailscale0 IPv4 (falls back to `tailscale ip`),
    otherwise pass the literal address/name through."""
    if host not in ("tailscale", "ts"):
        return host
    # primary: the interface address itself
    try:
        import subprocess
        ip = subprocess.run(["ip", "-4", "-o", "addr", "show", "tailscale0"],
                            capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", ip)
        if m:
            return m.group(1)
    except Exception:
        pass
    # fallback: `tailscale ip -4`
    try:
        import subprocess
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                            text=True, timeout=5).stdout.strip()
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
            return ip
    except Exception:
        pass
    raise SystemExit("no tailscale0 interface found; use --bind 0.0.0.0 or an "
                     "explicit address")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=DEF_METRICS)
    ap.add_argument("--log", default=DEF_LOG)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="single snapshot, exit")
    ap.add_argument("--serve-port", type=int, default=0, help="serve web/ + API on this port")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="bind address for --serve-port (default 127.0.0.1; "
                         "'tailscale'/'ts' auto-detects the tailscale0 IPv4; "
                         "'0.0.0.0' listens on all interfaces)")
    ap.add_argument("--from-start", action="store_true", help="read log from beginning")
    args = ap.parse_args(argv)

    tailer = LogTailer(args.log, from_start=args.from_start or args.once)
    hist = {"kv": deque(maxlen=180)}
    started = time.time()
    errors = 0

    if args.serve_port:
        bind = _resolve_bind(args.bind)
        srv = Serve(DEF_WEB, os.path.dirname(args.out), args.serve_port)
        httpd = ThreadingHTTPServer((bind, args.serve_port), srv.handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[collector] serving http://{bind}:{args.serve_port}/ (web: {DEF_WEB})", flush=True)

    if args.once:
        m = fetch_json(args.metrics)
        tailer.pump()
        snap = compose(m, tailer, started, hist)
        snap["sources"]["metrics"]["url"] = args.metrics
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(snap, f, indent=1)
        print(f"[collector] wrote {args.out} ({os.path.getsize(args.out)} B)", flush=True)
        return 0

    print(f"[collector] metrics={args.metrics} log={args.log} out={args.out}", flush=True)
    while True:
        try:
            m = fetch_json(args.metrics)
            errors = 0
        except Exception as e:
            m = None
            errors += 1
            if errors in (1, 10, 100):
                print(f"[collector] metrics poll failed ({errors}x): {e}", flush=True)
        tailer.pump()
        snap = compose(m, tailer, started, hist)
        snap["sources"]["metrics"]["url"] = args.metrics
        tmp = args.out + ".tmp"
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, args.out)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
