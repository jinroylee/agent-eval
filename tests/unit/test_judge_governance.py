"""Tests: gates refuse uncertified/stale judges; PPI corrects judge-scored CIs."""

import numpy as np
import pytest

from agent_eval.config.schema import ConfigModel, GateModel, JudgeModel, SuiteModel
from agent_eval.core.errors import UncertifiedJudge
from agent_eval.judges.certify import CertificationRecord
from agent_eval.judges.governance import assert_gated_judges_certified, judge_gate_ci
from agent_eval.judges.registry import JudgeRegistry


def _cfg(certification_ref="cert::j", gated=True):
    gate = GateModel(thresholds={"quality": 0.7}, require_pass=["quality"]) if gated else GateModel()
    return ConfigModel(
        agent_type="plain",
        judges={"j": JudgeModel(model="m", prompt_version="v1", rubric_id="r1",
                                certification_ref=certification_ref)},
        suites={"response": SuiteModel(
            metrics=[{"type": "judge", "name": "quality", "params": {"judge": "j"}}], gate=gate)},
    )


def _registry(passed=True, fingerprint="m|v1|r1"):
    reg = JudgeRegistry()
    reg.add_certification(
        CertificationRecord("cert::j", "j", fingerprint, 0.9 if passed else 0.1, 1.0, 10, passed)
    )
    return reg


def test_uncertified_judge_in_gate_raises():
    with pytest.raises(UncertifiedJudge):
        assert_gated_judges_certified(_cfg(), "response", JudgeRegistry())


def test_certified_judge_in_gate_passes():
    assert_gated_judges_certified(_cfg(), "response", _registry(passed=True))  # no raise


def test_stale_certification_raises():
    assert _cfg()  # fingerprint is m|v1|r1; cert below is for m|v9|r1
    with pytest.raises(UncertifiedJudge):
        assert_gated_judges_certified(_cfg(), "response", _registry(fingerprint="m|v9|r1"))


def test_failed_certification_raises():
    with pytest.raises(UncertifiedJudge):
        assert_gated_judges_certified(_cfg(), "response", _registry(passed=False))


def test_informational_judge_not_enforced():
    # Judge metric not referenced by the gate => certification not required.
    assert_gated_judges_certified(_cfg(certification_ref=None, gated=False), "response", JudgeRegistry())


def test_ppi_judge_ci_degrades_to_gold_under_noise():
    rng = np.random.default_rng(0)
    n, m = 60, 600
    y = rng.binomial(1, 0.6, n).astype(float)
    judge = rng.random(m)
    judge[:n] = rng.random(n)  # judge uncorrelated with truth
    theta, lo, hi = judge_gate_ci(judge, y, np.arange(n))
    se = np.std(y, ddof=1) / np.sqrt(n)
    gold_width = 2 * 1.959963984540054 * se
    assert abs(theta - y.mean()) < 0.05
    assert abs((hi - lo) - gold_width) < 0.02
