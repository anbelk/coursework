from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIGNATURE_PATH = ROOT / "figures" / "student_signature.png"
RUNTIME_PYTHON = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "python"
    / "bin"
    / "python3"
)


HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Подпись для титульного листа</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f4f5f7; color: #111827; }
    main { width: min(920px, calc(100vw - 32px)); }
    h1 { font-size: 22px; margin: 0 0 12px; font-weight: 650; }
    p { margin: 0 0 16px; color: #4b5563; }
    .pad { background: white; border: 1px solid #d1d5db; border-radius: 8px; box-shadow: 0 10px 30px rgba(15, 23, 42, .08); padding: 16px; }
    canvas { width: 100%; height: 280px; display: block; border: 1px dashed #9ca3af; border-radius: 6px; background: transparent; touch-action: none; }
    .actions { display: flex; gap: 10px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; font-size: 15px; cursor: pointer; }
    .primary { background: #111827; color: white; }
    .secondary { background: #e5e7eb; color: #111827; }
    #status { color: #374151; min-height: 22px; }
  </style>
</head>
<body>
  <main>
    <h1>Подпись для титульного листа</h1>
    <p>Нарисуй подпись в поле ниже. После сохранения файл <code>figures/student_signature.png</code> будет создан, а PDF пересобран.</p>
    <section class="pad">
      <canvas id="canvas" width="1600" height="520" aria-label="Поле для подписи"></canvas>
      <div class="actions">
        <button class="primary" id="save">Сохранить и пересобрать PDF</button>
        <button class="secondary" id="clear">Очистить</button>
        <span id="status"></span>
      </div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const status = document.getElementById("status");
    let drawing = false;
    let last = null;

    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";

    function point(event) {
      const rect = canvas.getBoundingClientRect();
      const touch = event.touches && event.touches[0];
      const source = touch || event;
      return {
        x: (source.clientX - rect.left) * (canvas.width / rect.width),
        y: (source.clientY - rect.top) * (canvas.height / rect.height)
      };
    }

    function start(event) {
      event.preventDefault();
      drawing = true;
      last = point(event);
    }

    function move(event) {
      if (!drawing) return;
      event.preventDefault();
      const next = point(event);
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(next.x, next.y);
      ctx.stroke();
      last = next;
    }

    function end(event) {
      event.preventDefault();
      drawing = false;
      last = null;
    }

    canvas.addEventListener("mousedown", start);
    canvas.addEventListener("mousemove", move);
    window.addEventListener("mouseup", end);
    canvas.addEventListener("touchstart", start, { passive: false });
    canvas.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", end, { passive: false });

    document.getElementById("clear").addEventListener("click", () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      status.textContent = "";
    });

    document.getElementById("save").addEventListener("click", async () => {
      status.textContent = "Сохраняю и пересобираю PDF...";
      const response = await fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: canvas.toDataURL("image/png") })
      });
      const result = await response.json();
      if (result.ok) {
        status.textContent = "Готово: подпись вставлена, build/main.pdf пересобран.";
      } else {
        status.textContent = "Ошибка: " + (result.error || "не удалось сохранить подпись");
      }
    });
  </script>
</body>
</html>
"""


class SignatureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self) -> None:
        if self.path != "/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image = payload["image"]
            prefix = "data:image/png;base64,"
            if not image.startswith(prefix):
                raise ValueError("unexpected image payload")
            SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SIGNATURE_PATH.write_bytes(base64.b64decode(image[len(prefix) :]))
            python = str(RUNTIME_PYTHON if RUNTIME_PYTHON.exists() else Path(sys.executable))
            subprocess.run([python, "scripts/build_title_pdf.py"], cwd=ROOT, check=True)
            subprocess.run(["latexmk", "-xelatex", "main.tex"], cwd=ROOT, check=True)
            shutil.copy2(ROOT / "build" / "main.pdf", ROOT / "main.pdf")
            self._json({"ok": True, "path": str(SIGNATURE_PATH)})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, status=500)

    def _json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local signature pad for the thesis title page")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SignatureHandler)
    print(f"Signature pad: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
