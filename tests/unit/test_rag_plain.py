"""Tests for RAG + Plain wiring: Recall@k gate from YAML, clustered SE, runtime critics."""

import json

from agent_eval.agents.plain.critic import build_plain_critic
from agent_eval.agents.rag.critic import build_rag_critic
from agent_eval.agents.rag.suites import build_search_suite
from agent_eval.config.loader import build_dataset_spec, load_config
from agent_eval.config.schema import ConfigModel, RuntimeCriticModel
from agent_eval.core.contracts import Decision, EvalContext, MetricResult, Tier
from agent_eval.core.gate import GatePolicy
from agent_eval.core.metric import BaseMetric
from agent_eval.core.suite import Suite
from agent_eval.datasets.base import load_dataset
from agent_eval.offline.runner import evaluate
from agent_eval.runtime.critic import critic_loop


class Graded(BaseMetric):
    name = "graded"
    tier = Tier.DETERMINISTIC

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, float(ctx.metadata["s"]), passed=None)


def _grouped(clustered: bool):
    data, idx = [], 0
    for s in (0.2, 0.5, 0.8):  # each "passage" group shares a score (strong within-cluster corr)
        for _ in range(4):
            cid = f"g{idx // 4}" if clustered else idx
            data.append(EvalContext("q", "o", metadata={"s": s, "cluster_id": cid}))
            idx += 1
    return data


def test_clustered_se_widens_ci_for_passage_grouped():
    def suite():
        return Suite("rag", "response", [Graded()], GatePolicy())

    clustered = evaluate(suite(), _grouped(True)).aggregates[0]
    singleton = evaluate(suite(), _grouped(False)).aggregates[0]
    cw = clustered.ci_high - clustered.ci_low
    sw = singleton.ci_high - singleton.ci_low
    assert cw > 1.5 * sw  # clustering on the unit of randomization inflates the SE


def test_rag_search_suite_builds_from_yaml_and_gates_on_recall(tmp_path):
    rows = [
        {"q": "a", "retrieved": ["d1", "d2", "x", "y"], "relevant": ["d1", "d2"]},  # recall 1.0
        {"q": "b", "retrieved": ["d3", "z", "w", "v"], "relevant": ["d3"]},  # recall 1.0
        {"q": "c", "retrieved": ["n1", "n2"], "relevant": ["d9"]},  # recall 0.0
    ]
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg_text = f"""
version: 1
agent_type: rag
datasets:
  ds:
    adapter: jsonl
    path: {tmp_path}/r.jsonl
    field_map: {{input: q, output: retrieved, expected: relevant}}
suites:
  search:
    dataset: ds
    metrics: [{{type: recall_at_k, name: recall_at_k, params: {{k: 4}}}}]
    gate: {{thresholds: {{recall_at_k: 0.5}}, require_pass: [recall_at_k]}}
"""
    (tmp_path / "rag.yaml").write_text(cfg_text)
    cfg = load_config(tmp_path / "rag.yaml")
    res = evaluate(build_search_suite(cfg), load_dataset(build_dataset_spec(cfg, "ds")))
    assert abs(res.aggregates[0].value - 2 / 3) < 1e-9
    assert res.verdict.passed is True  # mean recall 0.667 >= 0.5


def test_rag_critic_reretrieves_then_accepts():
    cfg = ConfigModel(
        agent_type="rag", runtime_critic=RuntimeCriticModel(tiers=["deterministic"], max_retries=2)
    )
    critic = build_rag_critic(cfg)
    relevant = ["d1", "d2"]
    retrievals = [["x1", "x2", "x3", "x4"], ["d1", "d2", "x", "y"]]  # insufficient, then good

    def gen(attempt, critique):
        return retrievals[min(attempt, 1)]

    res = critic_loop(gen, lambda ret: EvalContext("q", ret, expected=relevant), critic)
    assert res.decision is Decision.ACCEPT and res.attempts == 1


def test_plain_critic_escalates_on_low_consistency():
    cfg = ConfigModel(
        agent_type="plain",
        runtime_critic=RuntimeCriticModel(
            tiers=["uncertainty"], max_retries=1, tau={"selfcheck_consistency": 0.6}
        ),
    )
    critic = build_plain_critic(cfg)
    ctx = EvalContext("q", "paris", metadata={"samples": ["london", "berlin", "madrid"]})
    decision, assessment = critic.decide(ctx)
    assert decision is Decision.ESCALATE and not assessment.hard_fail
