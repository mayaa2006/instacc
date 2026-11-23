"""Run the agent: `python -m agent` (or start-agent.bat on Windows)."""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="instacc")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Use 0.0.0.0 or your Tailscale IP to reach it from your phone.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-bot", action="store_true", help="Don't start the Telegram bot")
    ap.add_argument("--ssl-certfile", help="HTTPS certificate (e.g. from `tailscale cert`): lets Android install the app")
    ap.add_argument("--ssl-keyfile", help="HTTPS private key matching --ssl-certfile")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from agent.config import load_settings

    settings = load_settings()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not settings.secrets.api_token:
        sys.exit("Refusing to listen on a non-local address without INSTACC_API_TOKEN set in .env")

    import uvicorn

    from agent.api.app import create_app

    app = create_app(settings, start_bot=not args.no_bot)
    ssl = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            sys.exit("--ssl-certfile and --ssl-keyfile must be given together")
        ssl = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile}
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(), **ssl)


if __name__ == "__main__":
    main()
