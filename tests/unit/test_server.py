"""API server tests — TestClient over ``create_app`` with deterministic judges.

The module skips entirely when the ``[server]`` extra is not installed, so the base suite is
unaffected. The default app fixture clears the judge env vars → the lexical stub, which makes
judge-metric scores exact and assertable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from agent_eval.server.catalog import ENDPOINTS, describe  # noqa: E402


# --------------------------------------------------------------------------- catalog table
def test_catalog_has_the_12_served_metrics():
    assert len(ENDPOINTS) == 12
    paths = [spec.path for spec in ENDPOINTS]
    assert len(set(paths)) == 12  # unique URLs
    # prefixed by agent-type family
    assert {spec.family for spec in ENDPOINTS} == {"common", "rag", "t2s"}
    # metadata-only metrics are deliberately NOT served
    served_types = {spec.type for spec in ENDPOINTS}
    assert "p95_latency" not in served_types and "token_usage" not in served_types


def test_catalog_t2s_paths_drop_the_redundant_prefix():
    by_path = {spec.path: spec for spec in ENDPOINTS}
    assert by_path["/t2s/faithfulness"].type == "t2s_faithfulness"
    assert by_path["/t2s/consistency"].type == "t2s_consistency"


def test_catalog_declares_params_and_judges():
    by_path = {spec.path: spec for spec in ENDPOINTS}
    assert by_path["/rag/recall_at_k"].params == ("k",)
    assert by_path["/common/llm_judge"].params == ("criteria", "scale")
    assert by_path["/t2s/soft_f1"].params == ("result_policy",)
    judge_paths = {s.path for s in ENDPOINTS if s.judge_based}
    assert judge_paths == {
        "/common/llm_judge", "/rag/faithfulness", "/rag/consistency",
        "/t2s/faithfulness", "/t2s/consistency",
    }


def test_catalog_types_and_requires_match_the_registry():
    from agent_eval.core.registry import default_registry

    reg = default_registry()
    types = set(reg.types())
    for spec in ENDPOINTS:
        if spec.extra == "t2s" and "soft_f1" not in types:
            continue  # [t2s] extra not installed — those types are legitimately absent
        assert spec.type in types, spec.path
        assert set(spec.requires) == reg.build(spec.type).requires, spec.path


def test_describe_renders_the_endpoint_contract():
    spec = next(s for s in ENDPOINTS if s.path == "/t2s/faithfulness")
    text = describe(spec)
    assert "`t2s_faithfulness`" in text
    assert "execution_result" in text
    assert "LLM judge" in text
    assert "[t2s]" in text


# --------------------------------------------------------------------------- wire schemas
def test_context_in_converts_to_frozen_eval_context():
    from agent_eval.core.contracts import EvalContext
    from agent_eval.server.schemas import EvalContextIn

    ctx = EvalContextIn(
        input="q", output="a", expected="ref",
        retrieved_context=["c1", "c2"], metadata={"retrieved_ids": ["d1"]},
    ).to_context()
    assert isinstance(ctx, EvalContext)
    assert ctx.input == "q" and ctx.output == "a" and ctx.expected == "ref"
    assert ctx.retrieved_context == ("c1", "c2")
    assert ctx.metadata == {"retrieved_ids": ["d1"]}
    # everything is optional on the wire — the metric's `requires` decides
    empty = EvalContextIn().to_context()
    assert empty.input is None and empty.output is None


def test_result_and_aggregate_mirror_core_contracts():
    from agent_eval.core.contracts import Cost, MetricResult
    from agent_eval.core.gate import MetricAggregate
    from agent_eval.server.schemas import AggregateOut, MetricResultOut

    out = MetricResultOut.from_result(
        MetricResult("m", 0.5, passed=True, confidence=0.9,
                     cost=Cost(latency_ms=1.5), detail={"reason": "r"}, error=None)
    )
    assert (out.metric, out.score, out.passed, out.confidence) == ("m", 0.5, True, 0.9)
    assert out.cost.latency_ms == 1.5 and out.detail == {"reason": "r"} and out.error is None

    from agent_eval.core.contracts import Aggregation

    agg = AggregateOut.from_aggregate(
        MetricAggregate("m", 0.8, 0.6, 0.9, 10, 2, Aggregation.MEAN, True)
    )
    assert (agg.metric, agg.value, agg.ci_low, agg.ci_high) == ("m", 0.8, 0.6, 0.9)
    assert (agg.n, agg.n_errors, agg.aggregation, agg.higher_is_better) == (10, 2, "mean", True)


# --------------------------------------------------------------------------- judge resolution
def test_judge_defaults_to_lexical_stub():
    from agent_eval.server import judge as judge_mod

    resolved = judge_mod.resolve_judge_from_env(environ={})
    assert resolved.kind == "stub"
    assert resolved.panel == ()
    # duck-typed JudgeBackend
    from agent_eval.judges.backend import JudgeRequest

    verdict = resolved.backend.evaluate(JudgeRequest(instruction="i", response="x", reference="x"))
    assert verdict.score == 1.0  # full token overlap with the reference


def test_judge_base_url_and_model_must_come_together():
    from agent_eval.server import judge as judge_mod

    with pytest.raises(RuntimeError, match="AGENT_EVAL_JUDGE_MODEL"):
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_BASE_URL": "http://x/v1"})
    with pytest.raises(RuntimeError, match="AGENT_EVAL_JUDGE_BASE_URL"):
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_MODEL": "m"})


def test_openai_compatible_judge_calls_chat_completions():
    import json

    import httpx

    from agent_eval.judges.backend import JudgeRequest
    from agent_eval.server import judge as judge_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "SCORE: 4\nREASON: solid"}}]}
        )

    resolved = judge_mod.resolve_judge_from_env(
        environ={
            "AGENT_EVAL_JUDGE_BASE_URL": "http://llm.test/v1",
            "AGENT_EVAL_JUDGE_MODEL": "test-model",
            "AGENT_EVAL_JUDGE_API_KEY": "sk-test",
        },
        transport=httpx.MockTransport(handler),
    )
    assert resolved.kind == "openai_compatible"
    assert "test-model" in resolved.detail

    verdict = resolved.backend.evaluate(JudgeRequest(instruction="rate", response="hi"))
    assert verdict.score == pytest.approx(0.75)  # SCORE: 4 on the default 1-5 scale
    assert verdict.reason == "solid"
    assert captured["url"] == "http://llm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["temperature"] == 0


def test_judge_factory_env_uses_the_yaml_factory_convention(tmp_path):
    from agent_eval.server import judge as judge_mod

    factory_file = tmp_path / "my_judges.py"
    factory_file.write_text(
        "from agent_eval.judges.backend import FunctionJudge\n"
        "def make_panel():\n"
        "    return [FunctionJudge(lambda r: 1.0), FunctionJudge(lambda r: 0.5)]\n",
        encoding="utf-8",
    )
    resolved = judge_mod.resolve_judge_from_env(
        environ={"AGENT_EVAL_JUDGE_FACTORY": f"{factory_file}:make_panel"}
    )
    assert resolved.kind == "factory"
    assert len(resolved.panel) == 1  # primary + 1 panel member


# --------------------------------------------------------------------------- app: health + catalog
@pytest.fixture()
def client(monkeypatch):
    """A TestClient with all judge env cleared → the deterministic lexical stub."""
    from fastapi.testclient import TestClient

    from agent_eval.server import judge as judge_mod
    from agent_eval.server.app import create_app

    for var in (
        judge_mod.ENV_FACTORY, judge_mod.ENV_BASE_URL, judge_mod.ENV_MODEL,
        judge_mod.ENV_API_KEY, judge_mod.ENV_TIMEOUT,
    ):
        monkeypatch.delenv(var, raising=False)
    return TestClient(create_app())


def test_health_reports_status_judge_and_availability(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["judge"]["kind"] == "stub"
    assert body["judge"]["panel_size"] == 1
    assert set(body["available"]) == {"t2s", "bertscore"}
    assert all(isinstance(v, bool) for v in body["available"].values())


def test_catalog_endpoint_lists_every_served_metric(client):
    entries = client.get("/catalog").json()
    assert len(entries) == 12
    by_path = {e["path"]: e for e in entries}
    assert by_path["/t2s/faithfulness"]["type"] == "t2s_faithfulness"
    assert by_path["/rag/recall_at_k"]["params"] == ["k"]
    assert by_path["/common/llm_judge"]["judge_based"] is True
    # sqlglot is installed in the dev env → t2s metrics are registered and available
    assert by_path["/t2s/soft_f1"]["available"] is True
    assert by_path["/t2s/soft_f1"]["aggregation"] == "mean"


def test_openapi_exposes_all_metric_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    from agent_eval.server.catalog import ENDPOINTS as _eps

    assert {spec.path for spec in _eps} <= paths


# --------------------------------------------------------------------------- metric endpoints
# Deterministic stub-judge expectations: lexical_overlap_judge scores the share of response
# tokens covered by (context ∪ reference) tokens.
RAG_IDS = {"retrieved_ids": ["d1", "d4", "d9"], "relevant_ids": ["d1", "d4"]}

HAPPY_CASES = [
    (
        "/common/llm_judge",
        {"input": "capital of France?", "output": "Paris", "expected": "Paris"},
        1.0,  # response fully covered by the reference
    ),
    ("/rag/recall_at_k", {"metadata": RAG_IDS}, 1.0),  # both gold ids in top-5
    ("/rag/precision_at_k", {"metadata": RAG_IDS}, 2 / 3),  # 2 relevant of 3 retrieved
    ("/rag/ndcg_at_k", {"metadata": RAG_IDS}, 1.0),  # relevant docs ranked first
    ("/rag/faithfulness", {"output": "alpha delta", "retrieved_context": ["alpha beta gamma"]}, 0.5),
    (
        "/rag/consistency",
        {"input": "q", "metadata": {"repeated_outputs": ["alpha beta", "alpha gamma"]}},
        0.5,  # one pair: |{alpha,beta} ∩ {alpha,gamma}| / 2
    ),
]

T2S_HAPPY_CASES = [
    (
        "/t2s/soft_f1",
        {"metadata": {
            "execution_result": [{"name": "Alice"}, {"name": "Carol"}],
            "gold_execution_result": [{"name": "Alice"}, {"name": "Carol"}],
        }},
        1.0,
    ),
    (
        "/t2s/component_match",
        {"metadata": {"sql": "SELECT name FROM employee", "gold_sql": "SELECT name FROM employee"}},
        1.0,
    ),
    ("/t2s/ast_valid", {"metadata": {"sql": "SELECT 1"}}, 1.0),
    (
        "/t2s/faithfulness",
        {"input": "q", "output": "3", "metadata": {"execution_result": [{"count": 3}]}},
        1.0,
    ),
    (
        "/t2s/consistency",
        {"input": "q", "metadata": {"repeated_sql": ["SELECT name FROM emp", "SELECT id FROM emp"]}},
        0.75,  # one pair: 3 of 4 response tokens shared
    ),
]


@pytest.mark.parametrize(("path", "ctx", "expected"), HAPPY_CASES)
def test_endpoint_scores_single_context(client, path, ctx, expected):
    resp = client.post(path, json={"contexts": [ctx]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["family"] == path.split("/")[1]
    assert body["n"] == 1 and body["n_errors"] == 0
    assert body["results"][0]["error"] is None
    assert body["results"][0]["score"] == pytest.approx(expected)
    assert body["aggregate"]["value"] == pytest.approx(expected)
    assert body["aggregate"]["n"] == 1


@pytest.mark.parametrize(("path", "ctx", "expected"), T2S_HAPPY_CASES)
def test_t2s_endpoint_scores_single_context(client, path, ctx, expected):
    pytest.importorskip("sqlglot")
    resp = client.post(path, json={"contexts": [ctx]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"][0]["error"] is None
    assert body["results"][0]["score"] == pytest.approx(expected)


def test_params_are_plumbed_into_the_metric(client):
    # k=1 → only d1 in the window → recall drops from 1.0 to 0.5
    body = client.post(
        "/rag/recall_at_k", json={"contexts": [{"metadata": RAG_IDS}], "params": {"k": 1}}
    ).json()
    assert body["results"][0]["score"] == pytest.approx(0.5)


def test_batch_aggregate_matches_the_public_evaluate(client):
    from agent_eval.core.contracts import EvalContext
    from agent_eval.core.gate import GatePolicy
    from agent_eval.core.registry import default_registry
    from agent_eval.core.suite import Suite
    from agent_eval.offline.runner import evaluate

    metas = [
        {"retrieved_ids": ["d1", "d4", "d9"], "relevant_ids": ["d1", "d4"]},  # 1.0
        {"retrieved_ids": ["d8", "d9"], "relevant_ids": ["d1", "d4"]},  # 0.0
        {"retrieved_ids": ["d1", "d9"], "relevant_ids": ["d1", "d4"]},  # 0.5
    ]
    body = client.post(
        "/rag/recall_at_k", json={"contexts": [{"metadata": m} for m in metas]}
    ).json()
    assert [r["score"] for r in body["results"]] == [
        pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.5),
    ]

    expected = evaluate(
        Suite("rag", "recall_at_k", [default_registry().build("recall_at_k")], GatePolicy()),
        [EvalContext(input=None, metadata=m) for m in metas],
    ).aggregates[0]
    agg = body["aggregate"]
    assert agg["value"] == pytest.approx(expected.value)
    assert agg["ci_low"] == pytest.approx(expected.ci_low)
    assert agg["ci_high"] == pytest.approx(expected.ci_high)
    assert (agg["n"], agg["aggregation"]) == (3, "mean")


def test_rate_metric_batch_uses_wilson_pass_rate(client):
    pytest.importorskip("sqlglot")
    sqls = ["SELECT 1", "SELECT 2", "SELECT FROM WHERE"]  # 2 valid, 1 broken
    body = client.post(
        "/t2s/ast_valid", json={"contexts": [{"metadata": {"sql": s}} for s in sqls]}
    ).json()
    assert [r["passed"] for r in body["results"]] == [True, True, False]
    assert body["aggregate"]["aggregation"] == "rate"
    from agent_eval.stats.intervals import wilson_interval

    value, lo, hi = wilson_interval(2, 3)
    assert body["aggregate"]["value"] == pytest.approx(value)
    assert body["aggregate"]["ci_low"] == pytest.approx(lo)
    assert body["aggregate"]["ci_high"] == pytest.approx(hi)


def test_consistency_scores_query_groups_of_repeated_runs(client):
    groups = [
        {"metadata": {"repeated_outputs": ["alpha beta", "alpha beta"]}},  # 1.0
        {"metadata": {"repeated_outputs": ["alpha beta", "alpha gamma"]}},  # 0.5
        {"metadata": {"repeated_outputs": ["delta one", "echo two"]}},  # 0.0
    ]
    body = client.post("/rag/consistency", json={"contexts": groups}).json()
    assert body["n"] == 3 and body["n_errors"] == 0
    assert [r["score"] for r in body["results"]] == [
        pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.0),
    ]
    assert body["results"][0]["detail"]["n_runs"] == 2
    assert body["aggregate"]["value"] == pytest.approx(0.5)


def test_t2s_consistency_no_longer_accepts_digest_params(client):
    pytest.importorskip("sqlglot")
    resp = client.post(
        "/t2s/consistency",
        json={"contexts": [{"metadata": {"repeated_sql": ["SELECT 1", "SELECT 1"]}}],
              "params": {"sample_rows": 3}},
    )
    assert resp.status_code == 422
    assert "sample_rows" in resp.json()["detail"]


def test_judge_metrics_score_each_item_exactly_once():
    from fastapi.testclient import TestClient

    from agent_eval.judges.backend import FunctionJudge
    from agent_eval.server.app import create_app

    calls = {"n": 0}

    def counting_judge(request):
        calls["n"] += 1
        return 1.0

    app = create_app(judge=FunctionJudge(counting_judge))
    body = TestClient(app).post(
        "/rag/faithfulness",
        json={"contexts": [{"output": "x", "retrieved_context": ["x"]} for _ in range(3)]},
    ).json()
    assert body["n"] == 3
    assert calls["n"] == 3  # replayed for aggregation, never re-scored


def test_scale_and_k_params_are_shape_checked(client):
    ok = client.post(
        "/common/llm_judge",
        json={"contexts": [{"input": "q", "output": "Paris", "expected": "Paris"}],
              "params": {"scale": [1, 10]}},
    )
    assert ok.status_code == 200  # a JSON list arrives as the (lo, hi) tuple the metric wants
    bad_scale = client.post(
        "/common/llm_judge",
        json={"contexts": [{"input": "q", "output": "a"}], "params": {"scale": [1, 2, 3]}},
    )
    assert bad_scale.status_code == 422
    assert "scale" in bad_scale.json()["detail"]
    bad_k = client.post(
        "/rag/recall_at_k",
        json={"contexts": [{"metadata": RAG_IDS}], "params": {"k": 0}},
    )
    assert bad_k.status_code == 422
    assert "'k'" in bad_k.json()["detail"]


def test_metric_route_operation_ids_are_unique(client):
    spec = client.get("/openapi.json").json()
    op_ids = [op["operationId"] for path in spec["paths"].values() for op in path.values()]
    assert len(op_ids) == len(set(op_ids))


# --------------------------------------------------------------------------- error model
def test_item_error_is_reported_not_scored(client):
    body = client.post(
        "/rag/recall_at_k",
        json={"contexts": [
            {"metadata": {"retrieved_ids": ["d1"], "relevant_ids": ["d1"]}},
            {"metadata": {}},  # missing ids → this item errors, the request does not
        ]},
    ).json()
    assert body["n"] == 2 and body["n_errors"] == 1
    assert body["results"][0]["error"] is None
    assert body["results"][1]["error"]
    # errored item excluded from the denominator — the aggregate is over the 1 valid item
    assert body["aggregate"]["n"] == 1
    assert body["aggregate"]["value"] == pytest.approx(1.0)


def test_all_items_errored_yields_empty_aggregate(client):
    body = client.post("/rag/recall_at_k", json={"contexts": [{"metadata": {}}]}).json()
    assert body["n_errors"] == 1
    assert body["aggregate"]["n"] == 0 and body["aggregate"]["value"] == 0.0


def test_empty_contexts_is_a_422(client):
    assert client.post("/rag/recall_at_k", json={"contexts": []}).status_code == 422


def test_unknown_param_is_a_422(client):
    resp = client.post(
        "/rag/recall_at_k",
        json={"contexts": [{"metadata": RAG_IDS}], "params": {"kk": 3}},
    )
    assert resp.status_code == 422
    assert "kk" in resp.json()["detail"]


def test_bad_param_value_is_a_422(client):
    resp = client.post(
        "/rag/recall_at_k",
        json={"contexts": [{"metadata": RAG_IDS}], "params": {"k": "not-a-number"}},
    )
    assert resp.status_code == 422


def test_null_param_means_default(client):
    # an explicit JSON null is "use the default", uniformly — not judge-dependent behavior
    import importlib.util

    cases = [
        ("/common/llm_judge",
         {"contexts": [{"input": "q", "output": "Paris", "expected": "Paris"}],
          "params": {"scale": None}}),
        ("/rag/recall_at_k",
         {"contexts": [{"metadata": RAG_IDS}], "params": {"k": None}}),
    ]
    if importlib.util.find_spec("sqlglot"):  # the third params idiom (`key in p`, t2s digest)
        cases.append(
            ("/t2s/faithfulness",
             {"contexts": [{"input": "q", "output": "3",
                            # 2 rows: a 1x1 result would take the single-value digest
                            # early-return and never reach the sample_rows param
                            "metadata": {"execution_result": [{"count": 3}, {"count": 4}]}}],
              "params": {"sample_rows": None}}),
        )
    for path, payload in cases:
        resp = client.post(path, json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["results"][0]["error"] is None
        # the DEFAULT behavior must be observed (all three fixtures score 1.0 under defaults),
        # not a valid-but-wrong sentinel like k=0
        assert body["results"][0]["score"] == pytest.approx(1.0)


def test_t2s_endpoints_501_when_extra_missing():
    from fastapi.testclient import TestClient

    from agent_eval.core.registry import MetricRegistry
    from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
    from agent_eval.metrics import common as common_metrics
    from agent_eval.metrics import rag as rag_metrics
    from agent_eval.server.app import create_app

    reg = MetricRegistry()  # a registry as it would look without sqlglot installed
    common_metrics.register(reg)
    rag_metrics.register(reg)
    client = TestClient(create_app(registry=reg, judge=FunctionJudge(lexical_overlap_judge)))

    resp = client.post("/t2s/soft_f1", json={"contexts": [{"metadata": {}}]})
    assert resp.status_code == 501
    assert "agent-eval[t2s]" in resp.json()["detail"]
    entry = next(e for e in client.get("/catalog").json() if e["path"] == "/t2s/soft_f1")
    assert entry["available"] is False and entry["unavailable_hint"]


def test_bertscore_501_when_package_missing(client, monkeypatch):
    from agent_eval.server import app as app_mod

    monkeypatch.setattr(app_mod, "_bertscore_available", lambda: False)
    resp = client.post("/common/bertscore", json={"contexts": [{"output": "a", "expected": "a"}]})
    assert resp.status_code == 501
    assert "bert-score" in resp.json()["detail"]


# --------------------------------------------------------------------------- judge through the app
def test_factory_env_drives_the_app_judge(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from agent_eval.server import judge as judge_mod
    from agent_eval.server.app import create_app

    factory_file = tmp_path / "my_judges.py"
    factory_file.write_text(
        "from agent_eval.judges.backend import FunctionJudge\n"
        "def make_panel():\n"
        "    return [FunctionJudge(lambda r: 1.0), FunctionJudge(lambda r: 0.5)]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(judge_mod.ENV_FACTORY, f"{factory_file}:make_panel")
    client = TestClient(create_app())

    info = client.get("/health").json()["judge"]
    assert info["kind"] == "factory" and info["panel_size"] == 2

    body = client.post(
        "/rag/faithfulness", json={"contexts": [{"output": "x", "retrieved_context": ["y"]}]}
    ).json()
    # PoLL panel: mean of (1.0, 0.5) with agreement 1 - spread
    assert body["results"][0]["score"] == pytest.approx(0.75)
    assert body["results"][0]["confidence"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- provider presets + .env
def test_provider_preset_builds_openai_compatible_judge():
    from agent_eval.server import judge as judge_mod

    r = judge_mod.resolve_judge_from_env(
        environ={"AGENT_EVAL_JUDGE_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}
    )
    assert r.kind == "openai_compatible"
    assert "api.anthropic.com" in r.detail and "provider=anthropic" in r.detail

    r = judge_mod.resolve_judge_from_env(
        environ={"AGENT_EVAL_JUDGE_PROVIDER": "openai", "OPENAI_API_KEY": "k"}
    )
    assert "api.openai.com" in r.detail and "gpt-4o-mini" in r.detail


def test_provider_preset_respects_model_override_and_needs_its_key():
    from agent_eval.server import judge as judge_mod

    r = judge_mod.resolve_judge_from_env(
        environ={
            "AGENT_EVAL_JUDGE_PROVIDER": "openai",
            "AGENT_EVAL_JUDGE_MODEL": "my-model",
            "OPENAI_API_KEY": "k",
        }
    )
    assert r.detail.startswith("my-model @ ")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_PROVIDER": "anthropic"})
    with pytest.raises(RuntimeError, match="anthropic"):  # error lists the valid providers
        judge_mod.resolve_judge_from_env(environ={"AGENT_EVAL_JUDGE_PROVIDER": "gemini"})


def test_explicit_base_url_wins_over_provider():
    from agent_eval.server import judge as judge_mod

    r = judge_mod.resolve_judge_from_env(
        environ={
            "AGENT_EVAL_JUDGE_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "k",
            "AGENT_EVAL_JUDGE_BASE_URL": "http://gateway.internal/v1",
            "AGENT_EVAL_JUDGE_MODEL": "internal-model",
        }
    )
    assert "gateway.internal" in r.detail and "provider=" not in r.detail


def test_load_env_file_setdefault_semantics(tmp_path, monkeypatch):
    import os

    from agent_eval.server import judge as judge_mod

    monkeypatch.setenv("AE_DEMO_EXISTING", "shell-wins")
    monkeypatch.delenv("AE_DEMO_NEW", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nAE_DEMO_EXISTING=file-value\nAE_DEMO_NEW=hello\nBROKEN LINE\n",
        encoding="utf-8",
    )
    assert judge_mod.load_env_file(env_file) is True
    assert os.environ["AE_DEMO_EXISTING"] == "shell-wins"  # already-set wins
    assert os.environ["AE_DEMO_NEW"] == "hello"
    monkeypatch.delenv("AE_DEMO_NEW")
    assert judge_mod.load_env_file(tmp_path / "missing.env") is False
