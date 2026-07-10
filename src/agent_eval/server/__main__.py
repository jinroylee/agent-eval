"""Serve the metric API: ``agent-eval-serve`` (or ``python -m agent_eval.server``).

Defaults to loopback; pass ``--host 0.0.0.0`` to expose (the Dockerfile does). The judge is
configured via environment variables — see agent_eval/server/judge.py and docs/api-server.md.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agent-eval-serve", description="Run the agent-eval metric API server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="uvicorn workers (default: 1)")
    args = parser.parse_args(argv)

    import uvicorn  # deferred so `--help` works even on a broken uvicorn install

    uvicorn.run(
        "agent_eval.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
