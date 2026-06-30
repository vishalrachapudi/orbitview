"""Launch the OrbitView web app and open it in a browser.

    python -m iss_tracker            # serve and open the browser
    python -m iss_tracker --no-open  # serve only (for headless/remote use)
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OrbitView — live 3D satellite tracker.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload (development).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/"

    if not args.no_open and not args.reload:
        # Open the browser shortly after the server has had time to bind.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  OrbitView → {url}\n  Press Ctrl+C to stop.\n")
    uvicorn.run(
        "iss_tracker.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
