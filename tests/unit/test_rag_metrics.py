"""RAG metrics: retrieval (recall/precision/ndcg, ids in metadata) + judge groundedness."""

import math

from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.metrics.rag import Faithfulness, NdcgAtK, PrecisionAtK, RecallAtK, ResponseConsistency


def _ctx(retrieved, relevant, **extra):
    md = {MetaKey.RETRIEVED_IDS: retrieved, MetaKey.RELEVANT_IDS: relevant, **extra}
    return EvalContext(input="q", metadata=md)


def test_recall_at_k():
    ctx = _ctx(["d1", "x", "d2"], ["d1", "d2"])
    assert RecallAtK(k=2).score(ctx).score == 0.5  # only d1 in top-2
    assert RecallAtK(k=3).score(ctx).score == 1.0


def test_precision_at_k():
    assert abs(PrecisionAtK(k=3).score(_ctx(["d1", "x", "d2"], ["d1", "d2"])).score - 2 / 3) < 1e-9


def test_ndcg_rewards_higher_ranking():
    high = NdcgAtK(k=3).score(_ctx(["d1", "x", "y"], ["d1"])).score  # relevant at rank 1
    low = NdcgAtK(k=3).score(_ctx(["x", "y", "d1"], ["d1"])).score  # relevant at rank 3
    assert high == 1.0 and low < high
    assert abs(low - (1 / math.log2(4)) / 1.0) < 1e-9


def test_ndcg_graded_relevance():
    ctx = _ctx(["d1", "d2"], ["d1", "d2"], **{MetaKey.RELEVANCE: {"d1": 3.0, "d2": 1.0}})
    assert NdcgAtK(k=2).score(ctx).score == 1.0  # already in ideal order


def test_retrieval_requires_ids():
    assert RecallAtK().score(EvalContext(input="q")).error  # no retrieved_ids
    assert RecallAtK().score(_ctx(["d1"], [])).error  # no relevant_ids


def test_faithfulness_and_consistency_use_context_and_output():
    judge = FunctionJudge(lexical_overlap_judge)
    grounded = EvalContext(input="q", output="Paris is the capital of France",
                           retrieved_context=("Paris is the capital of France.",))
    assert Faithfulness(judge).score(grounded).score == 1.0
    assert ResponseConsistency(judge).score(grounded).score == 1.0


def test_faithfulness_requires_context():
    judge = FunctionJudge(lexical_overlap_judge)
    assert Faithfulness(judge).score(EvalContext(input="q", output="a")).error  # no retrieved_context
