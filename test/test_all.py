"""Stdlib-only tests (python3 -m unittest discover test -v).

Covers: maya-written GGUF v3 metadata parsing (u32 KV-type writer quirk),
GLM STRP expert-profile decode (rank-ordered perfect-matching semantics),
native_experts offset table, dense tensor arithmetic, log_parser event
taxonomy for the maya/glm dialect (Strata dialect kept as legacy), and
collector.compose() snapshot shape, GPU power accounting and network
link-utilisation maths.

Run from repo root: python3 -m unittest discover -s test -v
"""
import collections, json, os, struct, sys, tempfile, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "collect"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from log_parser import parse_line  # noqa: E402
import build_atlas as BA           # noqa: E402

LIVE = os.path.join(ROOT, "live")
MANIFEST = os.path.join(LIVE, "atlas_manifest.json")
MAYA_LOG = os.path.expanduser("~/projects/maya/maya-maya-s-v2-iq2_xxs-256k.log")


class TestLogParser(unittest.TestCase):
    def test_maya_engine_lines(self):
        cases = [
            # glm stat line (multiline stat fragments may fall to `log`)
            ("glm stat: decode 87.51 ms/tok (11.43 tok/s) | vram hit 92.11% | "
             "ram fetches 0.11/tok | disk 23.44 reads/tok in 16.44 waits/tok, "
             "28.59 ms/tok | promotions 23.44/tok | cpu lane 0.00/tok in "
             "0.00 ms/tok | prompt 1.5 ms/tok | vram 8887/9595 ram 3/32",
             "stat"),
            ("glm pack: CUDA0 KV streaming: 32768 of 262144 positions of each "
             "of 11 DSA layers in VRAM (0.37 GB), the latent cache in 2.95 GB "
             "of pinned RAM", "pack_kv"),
            ("glm fast: CUDA0 tiers warm: 9493 experts (67.6 GB) in 8 s "
             "(8.69 GB/s)", "tiers_warm"),
            ("glm spec: 18024 speculative positions, 90.7% accepted",
             "spec_positions"),
            ("glm slots: took back a 26200-token conversation from disk (0.1 s); "
             "226 new tokens to read", "slot_take"),
            ("glm fast: the expert profile /home/x/data/expert-profile-glm.bin "
             "ranks 13248 (layer, expert) pairs", "profile_loaded"),
        ]
        for line, want in cases:
            ev = parse_line(line)
            self.assertIsNotNone(ev, f"no event for: {line}")
            self.assertEqual(ev["type"], want, f"{line!r} -> {ev}")

    def test_maya_request_lines(self):
        ev = parse_line("[maya] done: 1983 tokens in 166 s (12.9 tok/s) "
                        "(stop, cancel=False), expert cache 98.1% hit")
        self.assertEqual(ev["type"], "done"); self.assertAlmostEqual(ev["tok_s"], 12.9)
        self.assertAlmostEqual(ev["cache_pct"], 98.1)
        ev = parse_line("[maya] prompt cap: refused a 232,457-token prompt "
                        "(cap 65,536; msgs=30 tools=80 last='role':'preview')")
        self.assertEqual(ev["type"], "refused")
        self.assertIsNone(parse_line(""))
        fb = parse_line("random noise")
        self.assertEqual(fb["type"], "log")  # catch-all fallback keeps unmatched lines

    def test_legacy_strata_dialect_still_parses(self):
        ev = parse_line("strata mtp: draft layer loaded, 837 MiB of VRAM "
                        "(experts 675, dense 111), files read in 0.42 s (1,853 MiB/s)")
        self.assertEqual(ev["type"], "mtp_load")
        ev = parse_line("strata generate: pre-filled 5175 of 5175 slots from "
                        "the profile in 1.3 s (7078 MB/s); slot 0 verified")
        self.assertEqual(ev["type"], "expert_prefill")

    def test_all_patterns_run_on_real_log(self):
        if not os.path.exists(MAYA_LOG):
            self.skipTest("maya engine log absent")
        types = {}
        with open(MAYA_LOG, "rb") as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
                ev = parse_line(line)
                if not ev:
                    continue
                for e in (ev if isinstance(ev, list) else [ev]):
                    types[e["type"]] = types.get(e["type"], 0) + 1
        self.assertGreater(len(types), 12, f"only {types}")
        self.assertGreater(types.get("log", 0) + types.get("stat", 0), 0)


class TestAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MANIFEST):
            raise unittest.SkipTest("run tools/build_atlas.py first")
        with open(MANIFEST) as f:
            cls.a = json.load(f)

    def test_manifest_shape(self):
        d = self.a["dims"]
        self.assertEqual(d["layers"], 46); self.assertEqual(d["experts"], 288)
        self.assertEqual(d["top_k_experts"], 8); self.assertEqual(d["hidden"], 4096)
        self.assertEqual(d["vocab"], 154880); self.assertEqual(d["kv_dtype"], "int8")
        self.assertEqual(len(self.a["layers"]), 46)
        self.assertEqual(d["expert_total"], 42 * 288)  # layers 3..44

    def test_layer_kinds(self):
        kinds = [l["kind"] for l in self.a["layers"]]
        self.assertEqual(kinds[0], "dsa")       # 3 dense-lead blocks are DSA
        self.assertEqual(kinds[3], "mla")
        self.assertEqual(kinds[45], "draft")    # NextN draft block
        self.assertEqual(sum(k == "mla" for k in kinds), 11)
        self.assertEqual(sum(k == "dsa" for k in kinds), 34)
        self.assertTrue(all(l["moe"] for l in self.a["layers"][3:]))

    def test_expert_blob_arithmetic(self):
        eb = self.a["layers"][3]["expert_blob"]
        self.assertGreater(eb["gate_per_expert"], 0)
        # blob_bytes / the *_per_expert fields are per-expert byte counts
        self.assertEqual(eb["blob_bytes"],
                         eb["gate_per_expert"] + eb["up_per_expert"]
                         + eb["down_per_expert"])
        self.assertTrue(eb["shard"] in (None,) or "of-00003" in eb["shard"])
        self.assertIsNone(self.a["layers"][0]["expert_blob"])  # dense lead

    def test_profile_roundtrip(self):
        prof = BA.parse_expert_profile(BA.DEF_PROFILE)
        self.assertEqual(prof["entries"], 46 * 288)
        self.assertEqual(prof["hot_pairs"], 6624)   # half of 13,248 slots
        self.assertEqual(prof["cold_rows"], 6624)
        self.assertEqual(prof["top_experts"][0]["rank"], 0)
        self.assertEqual(prof["top_experts"][0]["partner"], [23, 133])
        self.assertEqual(prof["pairs_first8"][0], [14, 219, 23, 133])
        self.assertEqual(prof["heat"][14][219], 65535)
        self.assertEqual(prof["version"], 1)
        self.assertEqual(prof["layers"], 46); self.assertEqual(prof["experts"], 288)

    def test_gguf_rejects_bad_magic(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as t:
            t.write(b"JUNK" + b"\0" * 100); t.flush()
            with self.assertRaises(SystemExit):
                BA.parse_gguf_shard(t.name)

    def test_gguf_maya_writer_quirk(self):
        # synthetic minimal maya-style GGUF v3: u32 version, u64 counts, and
        # the maya writer's u32 KV *type* fields (standard GGUF uses u8).
        def s(b): return struct.pack("<Q", len(b)) + b.encode()
        buf = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
        buf += s("general.architecture") + struct.pack("<I", 8) + s("glm5-next")
        buf += s("glm5-next.block_count") + struct.pack("<I", 5) + struct.pack("<i", 46)
        with tempfile.NamedTemporaryFile(suffix=".gguf") as t:
            t.write(buf); t.flush()
            m = BA.parse_gguf_shard(t.name)["meta"]
            self.assertEqual(m["general.architecture"], "glm5-next")
            self.assertEqual(m["glm5-next.block_count"], 46)

    def test_nextn_and_drafter_inventory(self):
        if not MANIFEST or not os.path.exists(MANIFEST):
            self.skipTest("no manifest")
        with open("web/atlas_data.js") as f:  # smoke: ATLAS_META parses
            txt = f.read()
        i = txt.index("window.ATLAS_META")
        j = txt.index("};", i)
        self.assertIn("window.ATLAS_HEAT", txt[j:])
        nn = self.a["nextn"]
        self.assertEqual(len(nn["tensors"]), 29)
        self.assertIn("blk.45.nextn.eh_proj.weight",
                      [t["name"] for t in nn["tensors"]])


class TestCollector(unittest.TestCase):
    def test_compose_shape(self):
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        metrics = {
            "engine": {"expert_slots": 10012, "vram_slots": 10012,
                       "ram_slots": 63, "experts": 12096,
                       "expert_cache_mib": 68033, "kv": "int8",
                       "kv_ctx": 262144, "trained_context": 1048576,
                       "kv_resident": 32768, "speculation": ["DFlash2"]},
            "tiers": {"now": {"draft_acc": 98, "draft_off": 118,
                              "vram_used": 9896.0, "ram_used": 51.0,
                              "vram_hit": 0.964, "ram_fetch": 0.13,
                              "disk": 12.548, "promo": 12.212}},
            "requests": [{"prompt_tokens": 12000, "output_tokens": 900}],
            "totals": {"prompt_tokens": 100, "reused": 60},
            "gpu_map": [{"name": "NVIDIA GB10", "total": 127.85e9,
                         "free": 6.89e9, "drafter": 0.75e9,
                         "parts": {"pool": 71.34e9}, "layers": [0, 45],
                         "measured": "nvml"}],
        }
        class FakeT:
            path, counters = "/dev/null", {"done": 2}
            events = [{"type": "done", "t": "00:00:00", "cache_pct": 96.5}]
        out = collector.compose(metrics, FakeT(), time.time())
        self.assertEqual(out["engine"]["expert_slots"], 10012)
        self.assertAlmostEqual(out["derived"]["mtp_accept_rate"], 98 / 118, places=3)
        self.assertAlmostEqual(out["derived"]["prompt_hit_rate"], 0.6)
        self.assertEqual(out["derived"]["speculation"], ["DFlash2"])
        ex = out["derived"]["experts"]
        self.assertEqual(ex["per_token_structural"], 344)   # 43 layers x top-8
        self.assertEqual(ex["slots_total"], 12096)
        self.assertEqual(ex["last_request_hit_pct"], 96.5)
        self.assertEqual(out["kv"]["dtype"], "int8")
        self.assertEqual(out["kv"]["capacity_tokens"], 262144)
        self.assertEqual(out["kv"]["seq_tokens"], 12900)
        self.assertEqual(out["gpu"]["name"], "NVIDIA GB10")
        self.assertAlmostEqual(out["gpu"]["pool_gb"], 71.34, places=2)

    def test_power_block(self):
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        metrics = {
            "hardware": {"gpu_power": 40.0, "gpu_power_limit": 80.0},
            "history": {"gpu_power": [10.0, 40.0, 30.0, None]},
            "live": {"tok_s": 20.0},
            "tiers": {"now": {}},
        }
        class FakeT:
            path, counters, events = "/dev/null", {}, []
        p = collector.compose(metrics, FakeT(), time.time())["power"]
        self.assertEqual(p["gpu_w"], 40.0)
        self.assertEqual(p["peak_w"], 40.0)
        self.assertAlmostEqual(p["mean_w"], 80 / 3, places=2)
        self.assertEqual(p["util_pct"], 50.0)          # 40 W of an 80 W cap
        self.assertEqual(p["j_per_token"], 2.0)        # 40 W / 20 tok/s
        self.assertEqual(p["tok_per_wh"], 1800.0)      # 20 tok/s * 3600 / 40 W
        self.assertEqual(p["series"], [10.0, 40.0, 30.0])   # None dropped

    def test_network_utilisation(self):
        """rate = d(bytes)*8/dt; util% = rate / negotiated link bitrate."""
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        ns = collector.NetSampler.__new__(collector.NetSampler)
        ns.prev = {"eth9": (0, 0)}
        ns.prev_t = time.time() - 1.0
        ns.speeds = {"eth9": (1000.0, "sysfs", time.time())}
        ns.speed_ttl = 1e9
        ns.hist = {k: collections.deque(maxlen=4) for k in ("util", "rx_mbps", "tx_mbps")}
        # 12.5 MB in 1 s = 100 Mbit/s = 10 % of a 1 Gbit/s link
        ns._read_dev = staticmethod(lambda: {"eth9": (12_500_000, 1_250_000)})
        d_if, d_sys = collector._default_ifaces, collector._sysfs
        self.addCleanup(setattr, collector, "_default_ifaces", d_if)
        self.addCleanup(setattr, collector, "_sysfs", d_sys)
        collector._default_ifaces = lambda: ["eth9"]
        collector._sysfs = lambda n, a: {"operstate": "up", "carrier": "1"}.get(a)
        snap = ns.sample()
        i = snap["ifaces"][0]
        self.assertEqual(i["name"], "eth9")
        self.assertAlmostEqual(i["rx_mbps"], 100.0, delta=1.0)
        self.assertAlmostEqual(i["rx_pct"], 10.0, delta=0.2)
        self.assertAlmostEqual(i["tx_pct"], 1.0, delta=0.1)
        self.assertAlmostEqual(i["util_pct"], 10.0, delta=0.2)   # max(rx,tx)
        self.assertEqual(snap["primary"], "eth9")
        self.assertEqual(len(snap["history"]["util"]), 1)

    def test_network_skips_virtual_and_down_links(self):
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        for name in ("docker0", "br-f347fd986567", "veth1234"):
            self.assertTrue(collector.SKIP_IFACE.match(name), name)
        # lo is kept (engine API traffic rides it) but is never the primary link
        for name in ("lo", "eth0", "enP7s7", "wlP9s9", "tailscale0"):
            self.assertIsNone(collector.SKIP_IFACE.match(name), name)

    def test_network_loopback_never_primary(self):
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        ns = collector.NetSampler.__new__(collector.NetSampler)
        ns.prev = {"lo": (0, 0), "eth9": (0, 0)}
        ns.prev_t = time.time() - 1.0
        ns.speeds = {"lo": (None, None, time.time()),
                     "eth9": (1000.0, "sysfs", time.time())}
        ns.speed_ttl = 1e9
        ns.hist = {k: collections.deque(maxlen=4) for k in ("util", "rx_mbps", "tx_mbps")}
        d_if, d_sys = collector._default_ifaces, collector._sysfs
        self.addCleanup(setattr, collector, "_default_ifaces", d_if)
        self.addCleanup(setattr, collector, "_sysfs", d_sys)
        collector._default_ifaces = lambda: ["eth9"]
        collector._sysfs = lambda n, a: {"operstate": "up", "carrier": "1"}.get(a)
        # loopback busier than the NIC, yet the NIC stays primary
        ns._read_dev = staticmethod(lambda: {"lo": (10**8, 10**8), "eth9": (10**5, 0)})
        snap = ns.sample()
        self.assertEqual(snap["primary"], "eth9")
        lo = [i for i in snap["ifaces"] if i["name"] == "lo"][0]
        self.assertEqual(lo["kind"], "loopback")
        self.assertIsNone(lo["util_pct"])            # no link rate -> no %
        self.assertEqual(snap["ifaces"][-1]["name"], "lo")   # sorted last
        self.assertNotIn("lo", [i["name"] for i in snap["ifaces"][:1]])
        self.assertAlmostEqual(snap["total_rx_mbps"], 0.8, delta=0.05)  # lo excluded

    def test_live_snapshot_valid_json(self):
        p = os.path.join(LIVE, "telemetry.json")
        if not os.path.exists(p):
            self.skipTest("collector not run yet")
        with open(p) as f:
            d = json.load(f)
        self.assertIn("ts", d)
        self.assertIn("derived", d)


if __name__ == "__main__":
    unittest.main()
