"""Tests for judge primitives: backend, PoLL panel, pairwise swap-average, JudgeMetric."""

from agent_eval.core.contracts import EvalContext, Tier
from agent_eval.judges.backend import FunctionJudge
from agent_eval.judges.config import JudgeVerdict
from agent_eval.judges.panel import pairwise_swap_average, poll_pointwise
from agent_eval.metrics.judge_ import JudgeMetric


def test_function_judge_returns_verdict():
    b = FunctionJudge(lambda crit, ctx: 0.9 if "good" in str(ctx.output) else 0.2)
    v = b.evaluate("quality", EvalContext("q", "this is good"))
    assert isinstance(v, JudgeVerdict) and v.score == 0.9


def test_poll_pointwise_averages_with_agreement_confidence():
    judges = [FunctionJudge(lambda c, x, s=s: s) for s in (0.8, 0.6, 0.7)]
    score, conf = poll_pointwise(judges, "crit", EvalContext("q", "o"))
    assert abs(score - 0.7) < 1e-9
    assert conf < 1.0  # spread across panel lowers confidence


def test_pairwise_swap_average_handles_consistency_and_position_bias():
    def consistent(c, x, y):
        return 1 if x == "A" else -1  # A wins regardless of position

    assert pairwise_swap_average(consistent, "c", "A", "B") == 1

    def position_biased(c, x, y):
        return 1  # whoever is first "wins"

    assert pairwise_swap_average(position_biased, "c", "A", "B") == 0  # verdict flips -> tie


def test_judge_metric_pointwise_and_panel():
    b = FunctionJudge(lambda c, x: 0.85)
    m = JudgeMetric(backend=b, criteria="helpfulness")
    r = m.score(EvalContext("q", "ans"))
    assert m.tier is Tier.JUDGE and abs(r.score - 0.85) < 1e-9

    panel = [FunctionJudge(lambda c, x: 0.75), FunctionJudge(lambda c, x: 0.95)]
    rp = JudgeMetric(backend=b, criteria="helpfulness", panel=panel).score(EvalContext("q", "ans"))
    assert abs(rp.score - (0.85 + 0.75 + 0.95) / 3) < 1e-9  # PoLL average across families
