"""Tests for the pure RAG retrieval metrics (no LLM): Recall@k, Precision@k, RetrievalSufficiency."""

from agent_eval.core.contracts import EvalContext
from agent_eval.metrics.retrieval_ import (
    PrecisionAtK,
    RecallAtK,
    RetrievalSufficiency,
)


def _ctx(retrieved, relevant):
    return EvalContext(input="q", output=list(retrieved), expected=list(relevant))


def test_recall_at_k():
    assert abs(RecallAtK(k=4).score(_ctx(["a", "b", "c", "d"], ["a", "c", "e"])).score - 2 / 3) < 1e-9
    assert abs(RecallAtK(k=1).score(_ctx(["a", "b"], ["a", "c", "e"])).score - 1 / 3) < 1e-9


def test_precision_at_k():
    assert abs(PrecisionAtK(k=3).score(_ctx(["a", "b", "c", "d"], ["a", "c"])).score - 2 / 3) < 1e-9


def test_retrieval_sufficiency_gate():
    suf = RetrievalSufficiency(k=4, min_recall=0.5)
    assert suf.score(_ctx(["a", "c", "x", "y"], ["a", "c"])).passed is True  # recall 1.0
    bad = suf.score(_ctx(["x", "y", "z", "w"], ["a", "c"]))  # recall 0
    assert bad.passed is False and bad.detail["recall"] == 0.0


def test_missing_relevant_is_error():
    r = RecallAtK().score(EvalContext(input="q", output=["a"], expected=None))
    assert r.error is not None
