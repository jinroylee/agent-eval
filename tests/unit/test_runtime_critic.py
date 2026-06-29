"""Runtime critique foundation: cheap-first assessment, hard/soft decisions, the retry loop."""

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetricResult
from agent_eval.core.metric import BaseMetric
from agent_eval.runtime.critic import Critic, CriticPolicy, Decision, critic_loop
from agent_eval.runtime.fallback import default_fallbacks


class Parses(BaseMetric):
    """A FREE grounded binary check (stands in for ast_valid): output must contain no '('."""

    name = "parses"
    aggregation = Aggregation.RATE
    cost_class = CostClass.FREE

    def _compute(self, ctx):
        ok = "(" not in str(ctx.output)
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class Confident(BaseMetric):
    """An EXPENSIVE graded check (stands in for a judge): confidence from metadata['conf']."""

    name = "confident"
    cost_class = CostClass.EXPENSIVE
    calls = 0

    def _compute(self, ctx):
        Confident.calls += 1
        c = float(ctx.metadata.get("conf", 1.0))
        return MetricResult(self.name, c, confidence=c)


def _critic():
    return Critic([Confident(), Parses()], CriticPolicy(tau={"confident": 0.6}))


def test_accept_when_all_pass():
    d, a = _critic().decide(EvalContext(input="q", output="ok", metadata={"conf": 0.9}))
    assert d is Decision.ACCEPT and a.accepted


def test_hard_fail_retries_then_falls_back():
    c = _critic()
    bad = EvalContext(input="q", output="bad(", metadata={"conf": 0.9})
    assert c.decide(bad, attempt=0)[0] is Decision.RETRY
    assert c.decide(bad, attempt=2)[0] is Decision.FALLBACK  # retries exhausted


def test_soft_low_confidence_escalates():
    d, a = _critic().decide(EvalContext(input="q", output="ok", metadata={"conf": 0.3}))
    assert d is Decision.ESCALATE and not a.hard_fail


def test_cheap_first_short_circuits_expensive_judge():
    Confident.calls = 0
    # the FREE grounded check fails, so the EXPENSIVE judge must never run
    _critic().assess(EvalContext(input="q", output="bad(", metadata={"conf": 0.9}))
    assert Confident.calls == 0


def test_critic_loop_recovers_on_retry():
    c = Critic([Parses()], CriticPolicy(max_retries=2))
    # first attempt is bad, retry produces a clean output
    outputs = ["bad(", "fixed"]
    result = critic_loop(
        generate=lambda attempt, critique: outputs[attempt],
        build_ctx=lambda out: EvalContext(input="q", output=out),
        critic=c,
    )
    assert result.decision is Decision.ACCEPT and result.output == "fixed" and result.attempts == 1


def test_critic_loop_falls_back_via_strategy_when_stuck():
    c = Critic([Parses()], CriticPolicy(max_retries=2))
    result = critic_loop(
        generate=lambda attempt, critique: "always(bad",  # never improves
        build_ctx=lambda out: EvalContext(input="q", output=out),
        critic=c,
        fallback=default_fallbacks().get("abstain"),
    )
    assert result.decision is Decision.ABSTAIN
