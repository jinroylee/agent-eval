"""Common metrics: llm_judge (single + panel), bertscore (injected scorer), p95_latency, token_usage."""

import pytest

from agent_eval.core.contracts import Aggregation, EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, JudgeVerdict, lexical_overlap_judge
from agent_eval.metrics.common import BertScore, LLMJudgeMetric, P95Latency, TokenUsage


class _SpyJudge:
    """Records every JudgeRequest; returns a fixed verdict."""

    def __init__(self, score: float = 1.0):
        self.requests = []
        self.score = score

    def evaluate(self, request):
        self.requests.append(request)
        return JudgeVerdict(self.score, reason="spy")


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


def test_llm_judge_stamps_system_prompt_on_requests():
    spy = _SpyJudge()
    LLMJudgeMetric(spy, system_prompt="Persona.").score(EvalContext(input="q", output="a"))
    assert spy.requests and all(r.system_prompt == "Persona." for r in spy.requests)


def test_judge_metric_rejects_non_string_system_prompt():
    with pytest.raises(TypeError):
        LLMJudgeMetric(FunctionJudge(lambda r: 1.0), system_prompt=42)


def test_llm_judge_default_scores_each_criterion_separately():
    spy = _SpyJudge(score=0.8)
    r = LLMJudgeMetric(spy).score(EvalContext(input="q", output="a"))
    names = ["Completeness", "Clarity", "Usefulness", "Relevance", "Friendliness"]
    assert len(spy.requests) == 5  # one focused call per default criterion
    assert list(r.detail["criteria"]) == names
    assert all(name in req.instruction for name, req in zip(names, spy.requests, strict=True))
    assert r.score == pytest.approx(0.8)  # mean of equal per-criterion scores
    assert set(r.detail["reasons"]) == set(names)
    assert r.error is None


def test_llm_judge_criteria_string_keeps_single_holistic_call():
    spy = _SpyJudge(score=0.6)
    r = LLMJudgeMetric(spy, criteria="Rate overall quality.").score(EvalContext(input="q", output="a"))
    assert len(spy.requests) == 1
    assert spy.requests[0].instruction == "Rate overall quality."
    assert "criteria" not in r.detail and r.detail["reason"] == "spy"


def test_llm_judge_custom_criteria_list_and_descriptions():
    spy = _SpyJudge()
    r = LLMJudgeMetric(
        spy, criteria=["Accuracy", {"name": "Tone", "description": "polite, professional"}]
    ).score(EvalContext(input="q", output="a"))
    assert len(spy.requests) == 2
    assert "Accuracy" in spy.requests[0].instruction
    assert "Tone (polite, professional)" in spy.requests[1].instruction
    assert list(r.detail["criteria"]) == ["Accuracy", "Tone"]


def test_llm_judge_criteria_panel_reports_mean_agreement():
    m = LLMJudgeMetric(FunctionJudge(lambda r: 0.8), panel=[FunctionJudge(lambda r: 0.6)])
    r = m.score(EvalContext(input="q", output="a"))
    assert r.score == pytest.approx(0.7)
    assert r.confidence == pytest.approx(0.8)  # 1 - spread, identical for every criterion
    assert r.detail["panel"] == 2
    assert all(v == pytest.approx(0.8) for v in r.detail["agreement"].values())


def test_llm_judge_per_criterion_requests_carry_shared_fields():
    spy = _SpyJudge()
    LLMJudgeMetric(spy, scale=(1, 10), system_prompt="Persona.").score(
        EvalContext(input="q", output="a", expected="ref")
    )
    for req in spy.requests:
        assert req.scale == (1, 10) and req.system_prompt == "Persona."
        assert req.question == "q" and req.response == "a" and req.reference == "ref"


@pytest.mark.parametrize(
    "bad", [[], ["ok", ""], [{"description": "no name"}], ["dup", "dup"], 42, [3]]
)
def test_llm_judge_rejects_malformed_criteria(bad):
    with pytest.raises((TypeError, ValueError)):
        LLMJudgeMetric(FunctionJudge(lambda r: 1.0), criteria=bad)
