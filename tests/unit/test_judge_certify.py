"""Tests for judge certification + the judge registry (with fingerprint staleness)."""

from agent_eval.core.contracts import EvalContext
from agent_eval.judges.backend import FunctionJudge
from agent_eval.judges.certify import certify_judge
from agent_eval.judges.config import JudgeConfig
from agent_eval.judges.registry import JudgeRegistry

_GOOD_WORDS = ("great", "excellent", "good")


def _samples():
    data = [
        ("great answer", 1.0), ("excellent work", 1.0), ("good enough", 1.0),
        ("terrible", 0.0), ("awful", 0.0), ("bad output", 0.0),
    ]
    return [(EvalContext("q", o), h) for o, h in data]


# A judge that tracks the human labels (certifiable) and one that's uninformative (not).
GOOD = FunctionJudge(lambda c, ctx: 0.9 if any(w in str(ctx.output) for w in _GOOD_WORDS) else 0.1)
UNINFORMATIVE = FunctionJudge(lambda c, ctx: 0.5)


def test_good_judge_certifies():
    rec = certify_judge("g", GOOD, "quality", _samples(), fingerprint="m|v1|r1")
    assert rec.passed is True and rec.kappa >= 0.6


def test_uninformative_judge_fails_certification():
    rec = certify_judge("u", UNINFORMATIVE, "quality", _samples(), fingerprint="m|v1|r1")
    assert rec.passed is False


def test_registry_is_certified_and_detects_staleness():
    reg = JudgeRegistry()
    jc = JudgeConfig(id="g", model="m", prompt_version="v1", rubric_id="r1", certification_ref="cert::g")
    reg.register(jc)
    reg.add_certification(certify_judge("g", GOOD, "quality", _samples(), fingerprint=jc.fingerprint()))

    assert reg.is_certified("cert::g", jc.fingerprint()) is True

    stale = JudgeConfig(id="g", model="m", prompt_version="v2", rubric_id="r1")  # prompt changed
    assert reg.is_certified("cert::g", stale.fingerprint()) is False  # cert no longer matches
    assert reg.is_certified("missing", jc.fingerprint()) is False
