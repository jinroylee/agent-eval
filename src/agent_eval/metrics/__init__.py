"""The metric catalog, organized by agent type.

- :mod:`agent_eval.metrics.common` — apply to every agent (judge, BERTScore, latency, tokens)
- :mod:`agent_eval.metrics.rag` — retrieval quality + answer groundedness
- :mod:`agent_eval.metrics.t2s` — text-to-SQL execution/AST correctness + groundedness

Build a registry with every family via :func:`agent_eval.core.registry.default_registry`. Each
module exposes ``register(registry)`` to add just its types.
"""

from agent_eval.metrics import common, rag

__all__ = ["common", "rag", "register_all"]


def register_all(registry) -> None:
    """Register common + RAG metrics, and T2S if the optional ``sqlglot`` dependency is present."""
    common.register(registry)
    rag.register(registry)
    try:
        from agent_eval.metrics import t2s

        t2s.register(registry)
    except ImportError:
        pass
