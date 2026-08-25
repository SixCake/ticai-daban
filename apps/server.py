# -*- coding: utf-8 -*-
"""轻量本地服务: 看板静态页 + JSON API

GET /                 → web/dashboard.html
GET /api/live         → data/live/latest.json
GET /api/radar        → data/live/radar.json (预警雷达)
GET /api/review?date= → data/review/review_DATE.json (缺失则现场构建)
GET /api/dates        → 最近60个交易日列表

启动: python apps/server.py [port]  默认8765
"""
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from apps.review import build_review  # noqa: E402

WEB = ROOT / "web"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
            return super().do_GET()
        if parsed.path == "/api/live":
            f = DATA / "live" / "latest.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists() else '{"error":"no live data"}')
        if parsed.path == "/api/radar":
            f = DATA / "live" / "radar.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists() else '{"error":"no radar data"}')
        if parsed.path == "/api/review":
            date = parse_qs(parsed.query).get("date", [None])[0]
            if not date:
                return self._send_json('{"error":"date required"}', 400)
            f = DATA / "review" / f"review_{date}.json"
            if not f.exists():
                try:
                    snap = build_review(date)
                    if "error" not in snap:
                        f.parent.mkdir(exist_ok=True)
                        f.write_text(json.dumps(snap, ensure_ascii=False),
                                     encoding="utf-8")
                    else:
                        return self._send_json(
                            json.dumps(snap, ensure_ascii=False), 404)
                except Exception as e:
                    return self._send_json(
                            json.dumps({"error": str(e)}, ensure_ascii=False), 500)
            return self._send_json(f.read_text(encoding="utf-8"))
        if parsed.path == "/api/dates":
            import pandas as pd
            ev = pd.read_parquet(DATA / "events_enriched.parquet",
                                 columns=["trade_date"])
            dates = sorted(ev["trade_date"].unique())[-60:][::-1]
            return self._send_json(json.dumps({"dates": dates}))
        return super().do_GET()

    def _send_json(self, text: str, code: int = 200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # 静默


if __name__ == "__main__":
    print(f"看板服务: http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
