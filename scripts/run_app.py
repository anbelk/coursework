from __future__ import annotations

import argparse

import uvicorn

from common.config import load_yaml
from common.paths import CONFIGS


def main() -> int:
    cfg = load_yaml(CONFIGS / "app.yaml", {"host": "127.0.0.1", "port": 8000, "reload": False})
    parser = argparse.ArgumentParser(description="Run AI Coauthor web app")
    parser.add_argument("--host", default=str(cfg.get("host", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(cfg.get("port", 8000)))
    parser.add_argument("--reload", action="store_true", default=bool(cfg.get("reload", False)))
    args = parser.parse_args()
    uvicorn.run("web.backend.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
