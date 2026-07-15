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


def test_faithfulness_uses_context_and_output():
    judge = FunctionJudge(lexical_overlap_judge)
    grounded = EvalContext(input="q", output="Paris is the capital of France",
                           retrieved_context=("Paris is the capital of France.",))
    assert Faithfulness(judge).score(grounded).score == 1.0


# --- consistency: self-consistency across repeated runs of the same query --------------------
def _runs_ctx(runs):
    return EvalContext(input="q", metadata={MetaKey.REPEATED_OUTPUTS: runs})


def test_consistency_identical_runs_score_one():
    judge = FunctionJudge(lexical_overlap_judge)
    r = ResponseConsistency(judge).score(_runs_ctx(["the answer is 42"] * 3))
    assert r.error is None and r.score == 1.0
    assert r.detail["n_runs"] == 3 and len(r.detail["pair_scores"]) == 3


def test_consistency_judges_every_unordered_pair():
    calls = {"n": 0}

    def counting(request):
        calls["n"] += 1
        return 1.0

    ResponseConsistency(FunctionJudge(counting)).score(_runs_ctx(["a", "b", "c", "d"]))
    assert calls["n"] == 6  # C(4,2)


def test_consistency_partial_disagreement():
    judge = FunctionJudge(lexical_overlap_judge)
    # pairs: (0,1)=1.0, (0,2)=0.5, (1,2)=0.5 -> mean 2/3
    r = ResponseConsistency(judge).score(_runs_ctx(["alpha beta", "alpha beta", "alpha gamma"]))
    assert abs(r.score - 2 / 3) < 1e-9


def test_consistency_needs_at_least_two_runs():
    judge = FunctionJudge(lexical_overlap_judge)
    assert ResponseConsistency(judge).score(EvalContext(input="q")).error  # missing key
    assert ResponseConsistency(judge).score(_runs_ctx(["only one"])).error  # < 2 runs
    assert ResponseConsistency(judge).score(_runs_ctx("not a list")).error  # wrong type


def test_consistency_panel_reports_mean_agreement():
    panel_metric = ResponseConsistency(FunctionJudge(lambda r: 1.0), [FunctionJudge(lambda r: 0.5)])
    r = panel_metric.score(_runs_ctx(["x", "y"]))
    assert r.score == 0.75 and r.confidence == 0.5  # mean(1.0, 0.5); 1 - spread


def test_faithfulness_requires_context():
    judge = FunctionJudge(lexical_overlap_judge)
    assert Faithfulness(judge).score(EvalContext(input="q", output="a")).error  # no retrieved_context
