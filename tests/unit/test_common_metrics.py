"""Common metrics: llm_judge (single + panel), bertscore (injected scorer), p95_latency, token_usage."""

from agent_eval.core.contracts import Aggregation, EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.metrics.common import BertScore, LLMJudgeMetric, P95Latency, TokenUsage


def test_llm_judge_single_backend():
    m = LLMJudgeMetric(FunctionJudge(lexical_overlap_judge))
    ctx = EvalContext(input="capital of France?", output="Paris", expected="Paris")
    r = m.score(ctx)
    assert r.score > 0 and r.error is None


def test_llm_judge_panel_reports_confidence():
    m = LLMJudgeMetric(FunctionJudge(lambda req: 0.8), panel=[FunctionJudge(lambda req: 0.6)])
    r = m.score(EvalContext(input="q", output="a"))
    assert abs(r.score - 0.7) < 1e-9 and r.confidence is not None


def test_llm_judge_requires_input_and_output():
    assert LLMJudgeMetric(FunctionJudge(lambda req: 1.0)).score(EvalContext(input="q")).error


def test_bertscore_with_injected_scorer():
    m = BertScore(scorer=lambda cands, refs: [0.91])
    r = m.score(EvalContext(input="q", output="a paraphrase", expected="the reference"))
    assert abs(r.score - 0.91) < 1e-9 and m.aggregation is Aggregation.MEAN


def test_bertscore_requires_reference():
    m = BertScore(scorer=lambda c, r: [1.0])
    assert m.score(EvalContext(input="q", output="a")).error  # expected missing


def test_p95_latency_reads_metadata_and_is_lower_better():
    m = P95Latency()
    r = m.score(EvalContext(input="q", metadata={MetaKey.LATENCY_MS: 1200.0}))
    assert r.score == 1200.0 and m.aggregation is Aggregation.P95 and not m.higher_is_better


def test_p95_latency_errors_without_metadata():
    assert P95Latency().score(EvalContext(input="q")).error


def test_token_usage_reads_metadata():
    r = TokenUsage().score(EvalContext(input="q", metadata={MetaKey.TOKENS: 800}))
    assert r.score == 800.0 and not TokenUsage().higher_is_better
