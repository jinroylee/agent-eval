"""Tests for runtime hallucination signals over pre-sampled answers (no live LLM)."""

from agent_eval.core.contracts import EvalContext, Tier
from agent_eval.metrics.uncertainty_ import SelfCheckConsistency, SemanticEntropy


def _ctx(output, samples):
    return EvalContext(input="q", output=output, metadata={"samples": samples})


def test_selfcheck_high_when_samples_agree():
    m = SelfCheckConsistency(threshold=0.5)
    r = m.score(
        _ctx(
            "the capital of france is paris",
            ["paris is the capital of france", "capital of france is paris"],
        )
    )
    assert m.tier is Tier.UNCERTAINTY
    assert r.score >= 0.9 and r.confidence >= 0.9


def test_selfcheck_low_when_samples_disagree():
    r = SelfCheckConsistency(threshold=0.6).score(
        _ctx("the answer is paris", ["completely different text", "nothing alike here", "unrelated words"])
    )
    assert r.score <= 0.4


def test_selfcheck_no_samples_is_neutral():
    assert SelfCheckConsistency().score(_ctx("anything", [])).score == 1.0


def test_semantic_entropy_certain_when_all_same():
    r = SemanticEntropy(sim_threshold=0.5).score(_ctx("paris", ["paris", "paris", "paris"]))
    assert r.score >= 0.99  # one cluster => no entropy => full certainty


def test_semantic_entropy_uncertain_when_diverse():
    r = SemanticEntropy(sim_threshold=0.6).score(_ctx("paris", ["london", "berlin", "madrid"]))
    assert r.score <= 0.2  # four distinct meanings => max entropy => low certainty


def test_ragas_module_importable():
    # The lazy RAGAS adapter must import even without the optional dependency installed.
    from agent_eval.metrics.ragas_ import RagasMetric

    assert isinstance(RagasMetric, type)
