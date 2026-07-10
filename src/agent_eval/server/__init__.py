"""agent-eval metric API server (optional ``[server]`` extra). See docs/api-server.md."""

from agent_eval.server.app import create_app

__all__ = ["create_app"]
