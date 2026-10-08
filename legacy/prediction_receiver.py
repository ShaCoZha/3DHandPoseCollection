#!/usr/bin/env python3
import argparse
import json
import threading
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


MAX_REQUEST_BYTES = 1024 * 1024
EVENT_HISTORY_SIZE = 512


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMU Gesture Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #667085;
      --line: #d8dee8;
      --paper: rgba(255, 255, 255, 0.92);
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --live: #17a673;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 8% 0%, #eef6ff 0, transparent 30rem),
        radial-gradient(circle at 100% 100%, #f2eefc 0, transparent 32rem),
        #f6f8fb;
    }

    main {
      width: min(1500px, calc(100% - 40px));
      margin: 0 auto;
      padding: 30px 0 40px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0;
      font-size: clamp(25px, 3vw, 40px);
      letter-spacing: -0.035em;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 15px;
    }

    .connection {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
      padding: 9px 13px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #98a2b3;
    }

    .connection.online .dot {
      background: var(--live);
      box-shadow: 0 0 0 5px rgba(23, 166, 115, 0.12);
    }

    .card {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--paper);
      box-shadow: 0 16px 50px rgba(16, 24, 40, 0.07);
    }

    .latest {
      display: flex;
      align-items: center;
      padding: clamp(24px, 4vw, 46px);
      border-left: 7px solid var(--accent);
      min-height: 245px;
    }

    .eyebrow {
      margin: 0 0 12px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .gesture-stage {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      justify-content: flex-end;
      width: 100%;
      min-height: 165px;
      padding-top: 12px;
    }

    .gesture {
      position: relative;
      margin: 0;
      font-size: clamp(48px, 8vw, 108px);
      font-weight: 800;
      line-height: 0.98;
      letter-spacing: -0.055em;
      overflow-wrap: anywhere;
    }

    .gesture.waiting {
      color: #98a2b3;
      font-size: clamp(34px, 5vw, 68px);
    }
    .gesture.entered { animation: reveal 360ms ease-out; }

    .previous-gesture {
      min-height: 52px;
      margin: 0 0 5px;
      color: #98a2b3;
      font-size: clamp(26px, 4vw, 52px);
      font-weight: 700;
      line-height: 1;
      letter-spacing: -0.045em;
      pointer-events: none;
      opacity: 0.58;
      transform-origin: left bottom;
    }

    .previous-gesture:empty { visibility: hidden; }
    .previous-gesture.updated { animation: moveToPrevious 520ms ease-out; }

    @keyframes reveal {
      0% { opacity: 0.25; transform: translateY(9px) scale(0.985); }
      100% { opacity: 1; transform: none; }
    }

    @keyframes moveToPrevious {
      0% { opacity: 0.9; transform: translateY(35px) scale(1.18); }
      100% { opacity: 0.58; transform: translateY(0) scale(1); }
    }

    @media (max-width: 720px) {
      main { width: min(100% - 20px, 1500px); padding-top: 18px; }
      header { align-items: flex-start; }
      .subtitle { display: none; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Temporal–Spectral Gesture Recognition</h1>
        <p class="subtitle">HAR-5 pretrained Local-to-Global ModernTCN · Live on-device inference</p>
      </div>
      <div id="connection" class="connection">
        <span class="dot"></span><span id="connectionText">Connecting</span>
      </div>
    </header>

    <section class="card latest" aria-live="polite">
      <div style="width: 100%">
        <p class="eyebrow">Latest gesture</p>
        <div id="gestureStage" class="gesture-stage">
          <p id="previousGesture" class="previous-gesture"></p>
          <p id="gesture" class="gesture waiting">Waiting for gesture…</p>
        </div>
      </div>
    </section>
  </main>

  <script>
    const connection = document.getElementById('connection');
    const connectionText = document.getElementById('connectionText');
    const previousGesture = document.getElementById('previousGesture');
    const gesture = document.getElementById('gesture');
    const pendingEvents = [];
    const PRESENTATION_MS = 220;
    let lastEventId = __INITIAL_EVENT_ID__;
    let presenting = false;
    let lastGestureLabel = null;

    function displayLabel(value) {
      if (!value || value === '-') return '';
      return String(value).replaceAll('_', ' ');
    }

    function isNegative(value) {
      return displayLabel(value).trim().toLowerCase() === 'negative';
    }

    function transitionTo(value) {
      const nextLabel = displayLabel(value);
      if (!nextLabel || isNegative(nextLabel)) return;

      if (lastGestureLabel !== null) {
        previousGesture.textContent = lastGestureLabel;
        previousGesture.classList.remove('updated');
        void previousGesture.offsetWidth;
        previousGesture.classList.add('updated');
      }
      lastGestureLabel = nextLabel;

      gesture.textContent = nextLabel;
      gesture.classList.remove('waiting');
      gesture.classList.remove('entered');
      void gesture.offsetWidth;
      gesture.classList.add('entered');
    }

    function presentNext() {
      if (presenting || pendingEvents.length === 0) return;

      presenting = true;
      transitionTo(pendingEvents.shift().label);
      window.setTimeout(() => {
        presenting = false;
        presentNext();
      }, PRESENTATION_MS);
    }

    async function receiveEvents() {
      try {
        const response = await fetch(`/api/events?after=${lastEventId}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        connection.classList.add('online');
        connectionText.textContent = 'Live';

        for (const event of data.events) {
          if (event.event_id > lastEventId) {
            lastEventId = event.event_id;
            if (!isNegative(event.label)) pendingEvents.push(event);
          }
        }
        presentNext();
      } catch (error) {
        connection.classList.remove('online');
        connectionText.textContent = 'Reconnecting';
      } finally {
        window.setTimeout(receiveEvents, 60);
      }
    }

    receiveEvents();
  </script>
</body>
</html>
"""


class PredictionState:
    def __init__(self):
        self._lock = threading.Lock()
        self._events = deque(maxlen=EVENT_HISTORY_SIZE)
        self._latest = {
            "event_id": 0,
            "label": None,
            "confidence": None,
            "client_id": None,
            "received_at": None,
        }

    def update(self, client_id, payload):
        confidence = payload.get("confidence")
        if confidence is None:
            confidence = payload.get("score")

        with self._lock:
            self._latest = {
                "event_id": self._latest["event_id"] + 1,
                "label": payload.get("label", "-"),
                "confidence": confidence,
                "client_id": client_id,
                "received_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            }
            self._events.append(dict(self._latest))
            return dict(self._latest)

    def snapshot(self):
        with self._lock:
            return dict(self._latest)

    def events_after(self, event_id):
        with self._lock:
            return [dict(event) for event in self._events if event["event_id"] > event_id]


class PredictionReceiver(BaseHTTPRequestHandler):
    server_version = "PredictionReceiver/2.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            initial_event_id = self.server.prediction_state.snapshot()["event_id"]
            page = DASHBOARD_HTML.replace(
                "__INITIAL_EVENT_ID__",
                str(initial_event_id),
            )
            self.send_bytes(
                200,
                page.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        if path == "/api/events":
            query = parse_qs(parsed.query)
            try:
                after = max(0, int(query.get("after", ["0"])[0]))
            except (TypeError, ValueError):
                self.send_json(400, {"error": "Query parameter 'after' must be an integer"})
                return
            self.send_json(
                200,
                {"events": self.server.prediction_state.events_after(after)},
            )
            return

        if path == "/api/latest":
            self.send_json(200, self.server.prediction_state.snapshot())
            return

        if path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
            return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 2 or parts[0] != "prediction":
            self.send_json(404, {"error": "Expected POST /prediction/{clientID}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length"})
            return

        if content_length <= 0:
            self.send_json(400, {"error": "Request body is empty"})
            return
        if content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "Request body is too large"})
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_json(400, {"error": f"Bad JSON: {exc}"})
            return

        if not isinstance(payload, dict):
            self.send_json(400, {"error": "JSON payload must be an object"})
            return

        client_id = unquote(parts[1])
        latest = self.server.prediction_state.update(client_id, payload)
        print(
            f"{latest['label']}  client={client_id}  event={latest['event_id']}",
            flush=True,
        )
        self.send_json(200, {"ok": True, "event_id": latest["event_id"]})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def send_bytes(self, status, body, content_type, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(
        description="Receive Apple Watch gesture predictions and show a live demo dashboard."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/interface to bind. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind. Default: 8000",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard in the default browser after startup.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PredictionReceiver)
    server.prediction_state = PredictionState()

    browser_url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard: {browser_url}", flush=True)
    print(
        f"Prediction endpoint: http://{args.host}:{args.port}/prediction/{{clientID}}",
        flush=True,
    )
    print(
        "Set the iPhone app Receiver URL to this machine's LAN URL, "
        f"for example http://192.168.1.10:{args.port}",
        flush=True,
    )

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(browser_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping receiver.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
