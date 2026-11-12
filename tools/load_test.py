"""4 simulated sessions hammering glm-5.3-flash; sample atlas telemetry every 2s.
Usage: python3 tools/load_test.py [--duration 50] [--port 8009] [--atlas-host 127.0.0.1:8091]
Writes /tmp/load_samples.json and prints a summary; see README load-test section."""
import argparse, json, threading, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=50)
ap.add_argument("--port", type=int, default=8009)
ap.add_argument("--atlas-host", default="127.0.0.1:8091")
args = ap.parse_args()

PORT, KEY = args.port, "sk-local"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TEL = f"http://{args.atlas_host}/api/telemetry"

PROMPTS = [
    "Explain in detail how a transformer's grouped-query attention differs from multi-head attention, with concrete tensor shapes.",
    "Write a careful summary of how speculative decoding works when the drafter is a block-diffusion model.",
    "Describe the tradeoffs between IQ2_XXS and Q4_K_M quantization for MoE expert tensors, layer by layer.",
    "Walk through how an expert-cache hit-rate affects decode speed on a unified-memory GPU, with examples.",
    "Draft a short design note: what would a per-expert activation histogram tell us about routing collapse, and how should the HUD visualise it?",
    "Compare serving the same MoE from a page-able expert cache versus a pinned pool: latency profile, memory pressure, failure modes.",
]
stop = time.time() + args.duration
errors = counts = 0

def worker(i):
    global errors, counts
    msgs = [{"role":"system","content":"You are a concise expert."},
            {"role":"user","content": PROMPTS[i % len(PROMPTS)]}]
    body = json.dumps({"model":"glm-5.3-flash","messages":msgs,
                       "max_tokens":1200,"stream":True,"think":"medium"}).encode()
    while time.time() < stop:
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": f"Bearer {KEY}", "Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                for line in r:
                    if time.time() >= stop: break
                    if line.startswith(b"data:") and b"delta" in line: counts += 1
        except Exception as e:
            errors += 1; time.sleep(0.5)

threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(4)]
for t in threads: t.start()

samples = []
while time.time() < stop:
    try:
        tel = json.loads(urllib.request.urlopen(TEL, timeout=2).read())
        live, hw = tel.get("live", {}), tel.get("hardware", {})
        pw, net = tel.get("power", {}), tel.get("network", {})
        samples.append({
            "t": round(len(samples)*2, 1),
            "phase": live.get("phase"), "tok_s": live.get("tok_s"),
            "vram_hit": live.get("vram_hit_rate"), "ecache": live.get("expert_cache_hit"),
            "mtp": live.get("mtp_accept_rate"), "kv_seq": tel.get("kv", {}).get("seq_tokens"),
            "kv_res": tel.get("kv", {}).get("resident_tokens"), "gpu_util": hw.get("gpu_util"),
            "gpu_temp": hw.get("gpu_temp"), "cpu": hw.get("cpu"),
            "power_w": pw.get("gpu_w"), "j_per_tok": pw.get("j_per_token"),
            "net_iface": net.get("primary"), "net_link_mbit": net.get("link_mbit"),
            "net_util_pct": net.get("util_pct"),
            "net_rx_mbps": net.get("rx_mbps"), "net_tx_mbps": net.get("tx_mbps"),
        })
    except Exception: pass
    time.sleep(2)
for t in threads: t.join(timeout=5)
json.dump(samples, open("/tmp/load_samples.json","w"))
print(f"workers: 4 | stream chunks: {counts} | errors: {errors} | samples: {len(samples)}")
