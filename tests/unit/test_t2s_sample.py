"""Tests for the T2S sample dataset + suite/critic wiring."""

from agent_eval.agents.t2s.critic import build_t2s_critic
from agent_eval.agents.t2s.sample import GOLD, broken_sql, build_db
from agent_eval.agents.t2s.suites import t2s_registry
from agent_eval.config.schema import ConfigModel, RuntimeCriticModel
from agent_eval.core.contracts import Tier
from agent_eval.execution.harness import ExecutionHarness


def test_all_gold_queries_execute(tmp_path):
    db = str(tmp_path / "s.sqlite")
    build_db(db)
    h = ExecutionHarness()
    assert len(GOLD) >= 10
    for item in GOLD:
        r = h.run(item["gold_sql"], db)
        assert r.error is None, f"gold failed for {item['question']!r}: {r.error}"


def test_broken_sql_changes_some_results(tmp_path):
    db = str(tmp_path / "s.sqlite")
    build_db(db)
    h = ExecutionHarness()
    changed = 0
    for item in GOLD:
        ok, _, _ = h.execution_match(broken_sql(item["gold_sql"]), item["gold_sql"], db)
        if not ok:
            changed += 1
    assert changed >= 3  # the broken agent regresses several items


def test_t2s_registry_builds_execution_accuracy():
    m = t2s_registry().build({"type": "execution_accuracy", "name": "execution_accuracy"})
    assert m.name == "execution_accuracy"


def test_build_t2s_critic_from_config():
    cfg = ConfigModel(
        agent_type="t2s",
        runtime_critic=RuntimeCriticModel(tiers=["deterministic"], max_retries=2, tau={}),
    )
    critic = build_t2s_critic(cfg)
    assert critic.policy.max_retries == 2
    assert len(critic.metrics_by_tier[Tier.DETERMINISTIC]) >= 2
