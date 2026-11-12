"""Capture N seconds of the live atlas HUD as PNG frames via headless Chromium CDP.

Keeps ONE headless-chromium instance alive and screenshots it over the Chrome
DevTools Protocol (stdlib-only WebSocket client — no pip), so the cadence is
~1 fps instead of the ~0.2 fps of relaunching per frame (SwiftShader WebGL2
needs ~4 s of page load per launch).

Usage:
    python3 tools/screen_capture.py --duration 50 [--fps 1.0] [--out shots]
        [--url http://127.0.0.1:8091/] [--width 1680] [--height 945]

Writes shots/frame_0000.png …; print the frame count. Frames are assembled
into the README GIF by ffmpeg afterwards (see README "Demo capture").

Why SwiftShader flags: this box runs the collector headless (no X/Wayland
session in scope), so Chromium software-rendering must be forced:
    --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader
Verified by colour census (see TECHNIQUES §9 note): the render loop is alive
in the captured frames, not a dead black canvas.
"""
import argparse, base64, json, os, socket, struct, subprocess, time, urllib.request
import signal, sys

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=50)
ap.add_argument("--fps", type=float, default=1.0)
ap.add_argument("--out", default="shots")
ap.add_argument("--url", default="http://127.0.0.1:8091/")
ap.add_argument("--width", type=int, default=1680)
ap.add_argument("--height", type=int, default=945)
ap.add_argument("--port", type=int, default=9222)
ap.add_argument("--warmup", type=float, default=9.0, help="seconds to let the page load + WebGL settle")
args = ap.parse_args()

CHROMIUM = os.environ.get("CHROMIUM", "chromium-browser")


# ---------------------------------------------------------------- stdlib WS -
class WS:
    """Tiny RFC-6455 client, enough for CDP: text frames both ways, no
    fragmentation, server never sends big binary. Masks client frames."""
    def __init__(self, host, port, path):
        self.s = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.s.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.s.recv(4096)
        if b"101" not in resp.split(b"\r\n")[0]:
            raise RuntimeError("websocket handshake failed: " + resp[:200].decode())
        self.buf = b""

    def send(self, text):
        payload = text.encode()
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if len(payload) < 126:
            hdr = bytes([0x81, 0x80 | len(payload)])
        elif len(payload) < 65536:
            hdr = bytes([0x81, 0x80 | 126]) + struct.pack(">H", len(payload))
        else:
            hdr = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", len(payload))
        self.s.sendall(hdr + mask + masked)

    def recv(self):
        hdr = self._read(2)
        opcode, ln = hdr[0] & 0x0F, hdr[1] & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", self._read(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", self._read(8))[0]
        data = self._read(ln)
        if opcode == 8:
            raise ConnectionError("ws closed")
        if opcode in (9, 10):        # ping/pong control frames: skip
            return self.recv()
        return data.decode("utf-8", "replace")

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("ws EOF")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


# --------------------------------------------------------------- chrome ----
def find_target(port, tries=40):
    last = ""
    for i in range(tries):
        try:
            tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read())
            pages = [t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            if pages:
                return pages[0]["webSocketDebuggerUrl"]
        except Exception as e:
            last = e
        time.sleep(0.5)
    raise RuntimeError(f"no CDP page target after {tries} tries (last: {last})")


chrome_err = open("/tmp/chromium_cdp.err", "w")
proc = subprocess.Popen([
    CHROMIUM, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
    "--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader",
    f"--window-size={args.width},{args.height}", "--hide-scrollbars",
    f"--remote-debugging-port={args.port}", "--user-data-dir=/tmp/cdp-profile",
    args.url,
], stdout=subprocess.DEVNULL, stderr=chrome_err)
signal.signal(signal.SIGINT, signal.SIG_IGN)  # let the finally block clean up

os.makedirs(args.out, exist_ok=True)
try:
    ws_url = find_target(args.port)
    # webSocketDebuggerUrl looks like ws://127.0.0.1:9222/devtools/page/<id>
    ws_path = "/" + ws_url.split("/", 3)[3]
    ws = WS("127.0.0.1", args.port, ws_path)

    msg_id = 0
    def cdp(method, params=None):
        global msg_id
        msg_id += 1
        ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == msg_id:
                if "error" in r:
                    raise RuntimeError(r["error"]["message"])
                return r.get("result", {})
    cdp("Page.enable")
    time.sleep(args.warmup)          # atlas_data.js + first telemetry poll + GL warm

    frames, t_end, interval = 0, time.time() + args.duration, 1.0 / args.fps
    print(f"[capture] chromium up (pid {proc.pid}); warm {args.warmup}s; "
          f"capturing {args.duration}s @ {args.fps} fps -> {args.out}/", flush=True)
    t0 = time.time()
    while time.time() < t_end:
        try:
            shot = cdp("Page.captureScreenshot", {"format": "png"})
            data = base64.b64decode(shot["data"])
            name = os.path.join(args.out, f"frame_{frames:04d}.png")
            with open(name, "wb") as f:
                f.write(data)
            frames += 1
        except Exception as e:
            print(f"[capture] frame {frames}: {e}", flush=True)
            time.sleep(1)
        time.sleep(max(0, interval - (time.time() - t0) % interval))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
print(f"[capture] wrote {frames} frames -> {args.out}/frame_*.png", flush=True)
