"""Local training dashboard: py monitor_training.py [--run balanced_20260920]."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from weiqi.training_monitor import MonitorStore


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "weiqi" / "monitor_web"


def make_server(root, host="127.0.0.1", port=8766, preferred_run=None):
    store = MonitorStore(Path(root), preferred_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/runs":
                    self.send_json(store.runs())
                elif parsed.path == "/api/snapshot":
                    run_id = parse_qs(parsed.query).get("run", [None])[0]
                    self.send_json(store.snapshot(run_id))
                elif parsed.path == "/api/health":
                    self.send_json({"ok": True})
                elif parsed.path in {"/", "/index.html", "/style.css", "/app.js"}:
                    filename = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                    content_types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
                    data = (WEB / filename).read_bytes()
                    self.send_data(data, content_types[Path(filename).suffix] + "; charset=utf-8")
                elif parsed.path == "/favicon.ico":
                    self.send_data(b"", "image/x-icon", 204)
                else:
                    self.send_json({"error": "页面不存在"}, 404)
            except FileNotFoundError:
                self.send_json({"error": "找不到训练记录"}, 404)
            except ValueError:
                self.send_json({"error": "无效的训练记录路径或数据"}, 400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except OSError:
                self.send_json({"error": "暂时无法读取训练记录，请稍后重试"}, 503)

        def send_json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_data(data, "application/json; charset=utf-8", status)

        def send_data(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            # Successful 2-second polling should not fill the service log.
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.monitor_store = store
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="实时监控围棋训练胜率与对手（无需 PyTorch）")
    parser.add_argument("--root", type=Path, default=ROOT / "training_runs", help="训练记录的父目录")
    parser.add_argument("--run", help="页面默认选中的训练目录名")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    try:
        server = make_server(args.root, args.host, args.port, args.run)
    except OSError as error:
        parser.exit(1, f"监控服务启动失败：{error}\n")
    print(f"围棋训练监控：http://{args.host}:{server.server_port}\n读取目录：{args.root.resolve()}\n按 Ctrl+C 停止监控服务。", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
