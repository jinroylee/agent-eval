"""P1 acceptance: the T2S vertical slice end-to-end.

(a) offline gate PASSES on a good agent (returns gold) and FAILS on a broken one (drops WHERE),
(b) the regression gate flags the broken agent vs a frozen baseline,
(c) the runtime critic RETRYs then ACCEPTs once a valid query appears, and FALLs BACK when it never does,
(d) calibration writes tau and the runtime critic consumes it (offline -> runtime round-trip).
"""

import json

from agent_eval.agents.t2s.critic import build_t2s_critic
from agent_eval.agents.t2s.sample import GOLD, broken_sql, build_db
from agent_eval.agents.t2s.suites import t2s_registry
from agent_eval.config.loader import load_config
from agent_eval.config.schema import ConfigModel, RuntimeCriticModel
from agent_eval.core.contracts import Decision, EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.execution.harness import ExecutionHarness
from agent_eval.offline.calibrate import calibrate_and_write
from agent_eval.offline.regression import mcnemar_regression
from agent_eval.runtime.critic import critic_loop
from agent_eval.runtime.fallback import default_fallbacks


def _contexts(db, agent):
    return [
        EvalContext(
            input=it["question"], output=agent(it["gold_sql"]), expected=it["gold_sql"],
            metadata={"db_ref": db},
        )
        for it in GOLD
    ]


def _search_suite():
    reg = t2s_registry()
    return Suite(
        "t2s",
        "search",
        [reg.build("execution_accuracy")],
        GatePolicy(thresholds={"execution_accuracy": 0.8}, require_pass=["execution_accuracy"]),
    )


def test_offline_gate_passes_good_and_fails_broken(tmp_path):
    from agent_eval.offline.runner import evaluate

    db = build_db(str(tmp_path / "s.sqlite"))
    suite = _search_suite()

    good = evaluate(suite, _contexts(db, lambda g: g))
    assert good.verdict.passed is True
    assert good.aggregates[0].value == 1.0

    broken = evaluate(suite, _contexts(db, broken_sql))
    assert broken.verdict.passed is False
    assert broken.aggregates[0].value < 0.8


def test_regression_gate_flags_broken_vs_frozen(tmp_path):
    db = build_db(str(tmp_path / "s.sqlite"))
    h = ExecutionHarness()
    baseline = [h.execution_match(it["gold_sql"], it["gold_sql"], db)[0] for it in GOLD]
    candidate = [h.execution_match(broken_sql(it["gold_sql"]), it["gold_sql"], db)[0] for it in GOLD]
    r = mcnemar_regression(baseline, candidate, metric="execution_accuracy", min_effect=0.05)
    assert r.significant_drop is True


def test_runtime_critic_retries_then_accepts(tmp_path):
    db = build_db(str(tmp_path / "s.sqlite"))
    cfg = ConfigModel(
        agent_type="t2s",
        runtime_critic=RuntimeCriticModel(tiers=["deterministic"], max_retries=2, tau={}),
    )
    critic = build_t2s_critic(cfg)
    gold = "SELECT name FROM employee WHERE manager_id IS NULL"
    attempts = ["SELECT FROM oops", "SELECT name FROM nope", gold]  # unparseable, bad table, valid

    def gen(attempt, critique):
        return attempts[min(attempt, 2)]

    res = critic_loop(
        gen, lambda o: EvalContext(input="q", output=o, metadata={"db_ref": db}), critic
    )
    assert res.decision is Decision.ACCEPT and res.output == gold and res.attempts == 2


def test_runtime_critic_falls_back_when_never_valid(tmp_path):
    db = build_db(str(tmp_path / "s.sqlite"))
    cfg = ConfigModel(
        agent_type="t2s",
        runtime_critic=RuntimeCriticModel(
            tiers=["deterministic"], max_retries=2, tau={}, fallback="abstain"
        ),
    )
    critic = build_t2s_critic(cfg)

    def gen(attempt, critique):
        return f"SELECT name FROM missing_table_{attempt}"  # always references a missing table

    res = critic_loop(
        gen,
        lambda o: EvalContext(input="q", output=o, metadata={"db_ref": db}),
        critic,
        fallback=default_fallbacks().get("abstain"),
    )
    assert res.decision is Decision.ABSTAIN


def test_calibration_round_trip_offline_to_runtime(tmp_path):
    db = build_db(str(tmp_path / "s.sqlite"))
    rows = [
        {"question": it["question"], "pred": it["gold_sql"], "gold": it["gold_sql"], "db": db}
        for it in GOLD
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg_text = f"""
version: 1
agent_type: t2s
datasets:
  gold:
    adapter: jsonl
    path: {tmp_path}/d.jsonl
    field_map: {{input: question, output: pred, expected: gold, db_ref: db}}
suites:
  search:
    dataset: gold
    metrics: [{{type: execution_accuracy, name: execution_accuracy}}]
    gate: {{thresholds: {{execution_accuracy: 0.8}}, require_pass: [execution_accuracy]}}
runtime_critic: {{tiers: [deterministic], tau: {{}}}}
"""
    cfg_path = tmp_path / "t2s.yaml"
    cfg_path.write_text(cfg_text)

    tau = calibrate_and_write(cfg_path, "search", max_false_fail=0.05, registry=t2s_registry())
    assert tau["execution_accuracy"] == 1.0  # good agent => operating point at 1.0

    reloaded = load_config(cfg_path)
    assert reloaded.runtime_critic.tau["execution_accuracy"] == 1.0
    critic = build_t2s_critic(reloaded)
    assert critic.policy.tau["execution_accuracy"] == 1.0  # offline tau reached the runtime critic
