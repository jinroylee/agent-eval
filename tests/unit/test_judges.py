"""Judge backends and the PoLL panel."""

import pytest

from agent_eval.judges.backend import (
    FunctionJudge,
    JudgeRequest,
    JudgeVerdict,
    LLMJudge,
    lexical_overlap_judge,
)
from agent_eval.judges.panel import poll_pointwise

REQ = JudgeRequest(
    instruction="rate it", response="Paris is the capital",
    context=("Paris is the capital of France",),
)


def test_llm_judge_parses_score_and_reason():
    def complete(_prompt: str) -> str:
        return "SCORE: 4\nREASON: mostly correct"

    v = LLMJudge(complete).evaluate(JudgeRequest(instruction="x", response="y", scale=(1, 5)))
    assert v.score == 0.75 and v.reason == "mostly correct"  # (4-1)/(5-1)


def test_llm_judge_clamps_and_normalizes():
    v = LLMJudge(lambda p: "SCORE: 9").evaluate(JudgeRequest(instruction="x", response="y", scale=(1, 5)))
    assert v.score == 1.0  # clamped to top of scale


def test_llm_judge_raises_on_unparseable():
    with pytest.raises(ValueError):
        LLMJudge(lambda p: "no score here").evaluate(REQ)


def test_render_prompt_includes_fields():
    prompt = LLMJudge.render_prompt(REQ)
    assert "rate it" in prompt and "Paris is the capital of France" in prompt and "SCORE:" in prompt


def test_function_judge_wraps_float():
    assert FunctionJudge(lambda req: 0.5).evaluate(REQ).score == 0.5


def test_lexical_overlap_judge_rewards_supported_response():
    supported = lexical_overlap_judge(REQ).score
    hallucinated = lexical_overlap_judge(
        JudgeRequest(instruction="x", response="completely unrelated tokens xyz", context=("Paris France",))
    ).score
    assert supported > hallucinated


def test_poll_pointwise_averages_and_reports_agreement():
    j1 = FunctionJudge(lambda req: 0.8)
    j2 = FunctionJudge(lambda req: 0.6)
    mean, agreement = poll_pointwise([j1, j2], REQ)
    assert abs(mean - 0.7) < 1e-9 and abs(agreement - 0.8) < 1e-9  # 1 - (0.8-0.6)


def test_poll_pointwise_single_judge_full_agreement():
    mean, agreement = poll_pointwise([FunctionJudge(lambda req: JudgeVerdict(0.5))], REQ)
    assert mean == 0.5 and agreement == 1.0


def test_poll_empty_panel_raises():
    with pytest.raises(ValueError):
        poll_pointwise([], REQ)
