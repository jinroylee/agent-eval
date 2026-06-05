"""Tests for the tiered runtime critic: decide() outcomes, retry->fallback loop, loop guards."""

from agent_eval.core.contracts import Decision, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric
from agent_eval.runtime.critic import Critic, critic_loop
from agent_eval.runtime.fallback import default_fallbacks
from agent_eval.runtime.policy import CriticPolicy


class PassIfGood(BaseMetric):
    name = "is_good"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = ctx.output == "good"
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"hint": "want good"})


class LowConfidence(BaseMetric):
    name = "uncertain"
    tier = Tier.UNCERTAINTY
    requires = frozenset({"output"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, 0.3, passed=None, confidence=0.3)


def _critic(max_retries=2, tiers=(Tier.DETERMINISTIC,), metrics=None):
    metrics = metrics or {Tier.DETERMINISTIC: [PassIfGood()]}
    policy = CriticPolicy(tiers=list(tiers), tau={"uncertain": 0.7}, max_retries=max_retries)
    return Critic(metrics, policy)


def _ctx(o):
    return EvalContext(input="q", output=o)


def test_decide_accept_on_good():
    d, a = _critic().decide(_ctx("good"), attempt=0)
    assert d is Decision.ACCEPT and a.accepted


def test_decide_retry_then_fallback_by_attempt():
    c = _critic(max_retries=2)
    assert c.decide(_ctx("bad"), attempt=0)[0] is Decision.RETRY
    assert c.decide(_ctx("bad"), attempt=2)[0] is Decision.FALLBACK


def test_decide_escalates_on_low_confidence_not_hard_fail():
    c = _critic(
        tiers=(Tier.DETERMINISTIC, Tier.UNCERTAINTY),
        metrics={Tier.DETERMINISTIC: [PassIfGood()], Tier.UNCERTAINTY: [LowConfidence()]},
    )
    d, a = c.decide(_ctx("good"), attempt=0)  # passes deterministic, fails uncertainty gate
    assert d is Decision.ESCALATE and not a.hard_fail


def test_loop_retries_then_succeeds():
    c = _critic(max_retries=3)

    def gen(attempt, critique):
        return "good" if attempt >= 2 else f"bad{attempt}"

    res = critic_loop(gen, _ctx, c)
    assert res.decision is Decision.ACCEPT and res.output == "good" and res.attempts == 2


def test_loop_oscillation_triggers_fallback():
    c = _critic(max_retries=5)
    res = critic_loop(lambda a, cr: "bad", _ctx, c, fallback=default_fallbacks().get("abstain"))
    assert res.decision is Decision.ABSTAIN


def test_loop_exhausts_retries_to_fallback():
    c = _critic(max_retries=2)
    res = critic_loop(
        lambda a, cr: f"bad{a}", _ctx, c, fallback=default_fallbacks().get("abstain")
    )
    assert res.decision is Decision.ABSTAIN and res.attempts == 2


def test_critique_names_failing_metric():
    a = _critic().assess(_ctx("bad"))
    assert not a.accepted and "is_good" in a.critique
