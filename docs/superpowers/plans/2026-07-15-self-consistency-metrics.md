# Self-Consistency Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redefine `consistency` (RAG) and `t2s_consistency` as LLM-judged self-consistency across N repeated runs of the same query (pairwise-mean), with `prediction.n_runs` harness support, and ripple through server catalog, tests, docs (EN+KR), and notebooks.

**Architecture:** Runs live in one `EvalContext`'s metadata arrays (`repeated_outputs`/`repeated_sql` — two new `MetaKey`s); a shared `SelfConsistencyMetric(JudgeMetric)` base in `metrics/common.py` judges every unordered pair and averages. Runner, server handler, schemas, and judge resolution are untouched; only the two catalog rows change server-side.

**Tech Stack:** existing framework modules; `itertools.combinations`; nbformat for notebook patching.

**Spec:** `docs/superpowers/specs/2026-07-15-self-consistency-metrics-design.md`

**Standing constraint:** NO git commits — the user reviews the working tree (skip every commit step). All stub-judge expectations below are exact: stub = token-recall of `response` against `reference` tokens, tokens = lowercase `[a-z0-9]+` (Korean-only strings tokenize to ∅ → keep ASCII/numeric tokens in fixtures).

---

### Task 1: Metrics core — contracts, shared base, RAG, T2S (+ unit tests)

**Files:**
- Modify: `src/agent_eval/core/contracts.py` (MetaKey block, ~line 50)
- Modify: `src/agent_eval/metrics/common.py`
- Modify: `src/agent_eval/metrics/rag.py`
- Modify: `src/agent_eval/metrics/t2s.py`
- Test: `tests/unit/test_rag_metrics.py`, `tests/unit/test_t2s_metrics.py`

- [ ] **Step 1: Write the failing tests.**

In `tests/unit/test_rag_metrics.py`, REPLACE `test_faithfulness_and_consistency_use_context_and_output` (lines 42-47) with:

```python
def test_faithfulness_uses_context_and_output():
    judge = FunctionJudge(lexical_overlap_judge)
    grounded = EvalContext(input="q", output="Paris is the capital of France",
                           retrieved_context=("Paris is the capital of France.",))
    assert Faithfulness(judge).score(grounded).score == 1.0


# --- consistency: self-consistency across repeated runs of the same query --------------------
def _runs_ctx(runs):
    return EvalContext(input="q", metadata={MetaKey.REPEATED_OUTPUTS: runs})


def test_consistency_identical_runs_score_one():
    judge = FunctionJudge(lexical_overlap_judge)
    r = ResponseConsistency(judge).score(_runs_ctx(["the answer is 42"] * 3))
    assert r.error is None and r.score == 1.0
    assert r.detail["n_runs"] == 3 and len(r.detail["pair_scores"]) == 3


def test_consistency_judges_every_unordered_pair():
    calls = {"n": 0}

    def counting(request):
        calls["n"] += 1
        return 1.0

    ResponseConsistency(FunctionJudge(counting)).score(_runs_ctx(["a", "b", "c", "d"]))
    assert calls["n"] == 6  # C(4,2)


def test_consistency_partial_disagreement():
    judge = FunctionJudge(lexical_overlap_judge)
    # pairs: (0,1)=1.0, (0,2)=0.5, (1,2)=0.5 -> mean 2/3
    r = ResponseConsistency(judge).score(_runs_ctx(["alpha beta", "alpha beta", "alpha gamma"]))
    assert abs(r.score - 2 / 3) < 1e-9


def test_consistency_needs_at_least_two_runs():
    judge = FunctionJudge(lexical_overlap_judge)
    assert ResponseConsistency(judge).score(EvalContext(input="q")).error  # missing key
    assert ResponseConsistency(judge).score(_runs_ctx(["only one"])).error  # < 2 runs
    assert ResponseConsistency(judge).score(_runs_ctx("not a list")).error  # wrong type


def test_consistency_panel_reports_mean_agreement():
    panel_metric = ResponseConsistency(FunctionJudge(lambda r: 1.0), [FunctionJudge(lambda r: 0.5)])
    r = panel_metric.score(_runs_ctx(["x", "y"]))
    assert r.score == 0.75 and r.confidence == 0.5  # mean(1.0, 0.5); 1 - spread
```

In `tests/unit/test_t2s_metrics.py`, REPLACE `test_t2s_consistency_judges_answer_against_result` (lines 177-182) with:

```python
def test_t2s_consistency_identical_sql_scores_one():
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx(**{MetaKey.REPEATED_SQL: ["SELECT name FROM emp", "SELECT name FROM emp"]})
    r = T2SConsistency(judge).score(ctx)
    assert r.error is None and r.score == 1.0


def test_t2s_consistency_divergent_sql_scores_low():
    judge = FunctionJudge(lexical_overlap_judge)
    # stub: |{select,name,from,emp} ∩ {select,id,from,emp}| / 4 = 0.75
    ctx = _ctx(**{MetaKey.REPEATED_SQL: ["SELECT name FROM emp", "SELECT id FROM emp"]})
    assert T2SConsistency(judge).score(ctx).score == 0.75


def test_t2s_consistency_needs_repeated_sql():
    judge = FunctionJudge(lexical_overlap_judge)
    assert T2SConsistency(judge).score(_ctx(output="something")).error  # no repeated_sql
    assert T2SConsistency(judge).score(_ctx(**{MetaKey.REPEATED_SQL: ["one"]})).error
```

(The old test asserted output-vs-execution-result judging; `test_t2s_judge_errors_without_result` at lines 185-188 stays — it tests `T2SFaithfulness`, unchanged.)

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/unit/test_rag_metrics.py tests/unit/test_t2s_metrics.py`
Expected: FAIL — `AttributeError: ... no attribute 'REPEATED_OUTPUTS'` / `REPEATED_SQL` (MetaKey missing), old-semantics classes still in place.

- [ ] **Step 3: Contracts.** In `src/agent_eval/core/contracts.py`, inside `MetaKey` after the `GOLD_EXECUTION_RESULT` line add:

```python
    REPEATED_OUTPUTS = "repeated_outputs"  # consistency: the N final responses to the SAME input (one per run)
    REPEATED_SQL = "repeated_sql"  # T2S consistency: the N generated SQLs for the SAME input (one per run)
```

and change the `retrieved_context` field comment in `EvalContext` from `# RAG: retrieved chunk texts (for faithfulness/consistency)` to `# RAG: retrieved chunk texts (for faithfulness)`.

- [ ] **Step 4: Shared base.** In `src/agent_eval/metrics/common.py`: add `from itertools import combinations` to the imports (stdlib group). After the `JudgeMetric` class add:

```python
def _repeated_runs(ctx: EvalContext, key: str) -> list[str]:
    runs = ctx.metadata.get(key)
    if runs is None:
        raise ValueError(f"metadata['{key}'] (the repeated runs for this query) is required")
    if isinstance(runs, str) or not isinstance(runs, Sequence):
        raise ValueError(f"metadata['{key}'] must be a list of runs (one generation per run)")
    if len(runs) < 2:
        raise ValueError(f"metadata['{key}'] needs at least 2 runs to measure consistency")
    return [str(r) for r in runs]


class SelfConsistencyMetric(JudgeMetric):
    """Pairwise self-consistency across repeated runs of the same query.

    Subclasses set ``runs_key`` (which metadata array holds the runs) and ``_rubric``. Every
    unordered pair (i < j) is judged for semantic equivalence — the pair rides in
    ``response``/``reference`` — and the score is the mean over the N(N-1)/2 pairs (a PoLL panel
    applies per pair; ``confidence`` is the mean panel agreement). Judge-call cost is quadratic
    in the number of runs.
    """

    requires = frozenset()  # runs live in metadata; checked in _compute
    runs_key: str = ""
    _rubric: str = ""

    def _compute(self, ctx: EvalContext) -> MetricResult:
        runs = _repeated_runs(ctx, self.runs_key)
        judges = [self.backend, *self.panel]
        pair_scores: list[float] = []
        agreements: list[float] = []
        for i, j in combinations(range(len(runs)), 2):
            request = JudgeRequest(
                instruction=self._rubric, question=str(ctx.input),
                response=runs[i], reference=runs[j],
            )
            if len(judges) > 1:
                score, agreement = poll_pointwise(judges, request)
                agreements.append(agreement)
            else:
                score = self.backend.evaluate(request).score
            pair_scores.append(score)
        value = sum(pair_scores) / len(pair_scores)
        confidence = sum(agreements) / len(agreements) if agreements else None
        return MetricResult(
            self.name, value, confidence=confidence,
            detail={"n_runs": len(runs), "pair_scores": [round(s, 4) for s in pair_scores]},
        )
```

Update the module docstring's metric table sentence if it enumerates judge consumers ("plus faithfulness/consistency in the RAG and T2S modules" stays accurate — no change needed there).

- [ ] **Step 5: RAG metric.** In `src/agent_eval/metrics/rag.py`:
  1. Import line: `from agent_eval.metrics.common import JudgeMetric, SelfConsistencyMetric, resolve_judge`.
  2. Delete `_CONSISTENCY_RUBRIC` (lines 104-108) and the old `ResponseConsistency(_ContextJudgeMetric)` class (lines 130-132). `_ContextJudgeMetric` and `Faithfulness` stay.
  3. Add after `Faithfulness`:

```python
# --------------------------------------------------------------------------- self-consistency (non-GT, judge)
_SELF_CONSISTENCY_RUBRIC = (
    "The RESPONSE TO EVALUATE and the REFERENCE ANSWER are two answers the same agent produced "
    "for the SAME user question on different runs. Judge whether they are consistent with each "
    "other: do they give the same substantive answer — the same facts, figures, and conclusion? "
    "Ignore differences in wording, order, or level of detail; penalize factual disagreement or "
    "contradictory conclusions."
)


class ResponseConsistency(SelfConsistencyMetric):
    """Self-consistency: do N answers to the SAME query (``metadata['repeated_outputs']``) agree?"""

    name = "consistency"
    runs_key = MetaKey.REPEATED_OUTPUTS
    _rubric = _SELF_CONSISTENCY_RUBRIC
```

  4. Module docstring table row for consistency becomes: `| ``consistency``   | no  | ``metadata['repeated_outputs']`` (+ ``input``)       |` and the intro sentence about faithfulness/consistency sharing `retrieved_context` is reworded (faithfulness reads the chunks; consistency compares repeated runs).
  5. Registration line unchanged: `registry.register("consistency", lambda p, ctx: ResponseConsistency(resolve_judge(ctx), ctx.panel))`.

- [ ] **Step 6: T2S metric.** In `src/agent_eval/metrics/t2s.py`:
  1. Import line: `from agent_eval.metrics.common import JudgeMetric, SelfConsistencyMetric, resolve_judge`.
  2. Delete `_T2S_CONSISTENCY_RUBRIC` (lines 337-344) and `class T2SConsistency(_ExecutionJudgeMetric)` (lines 392-394). `_ExecutionJudgeMetric` and `T2SFaithfulness` stay.
  3. Add after `T2SFaithfulness`:

```python
# --------------------------------------------------------------------------- self-consistency (non-GT, judge)
_SQL_CONSISTENCY_RUBRIC = (
    "The RESPONSE TO EVALUATE and the REFERENCE ANSWER are two SQL queries the same agent "
    "generated for the SAME user question on different runs. Judge whether they are semantically "
    "equivalent: run against the same database, would they return the same result (same rows, "
    "columns, filters, grouping, and aggregation)? Ignore formatting, letter case, aliases, and "
    "quoting; penalize different tables, columns, filters, aggregations, or limits."
)


class T2SConsistency(SelfConsistencyMetric):
    """Self-consistency: do N generated SQLs for the SAME query (``metadata['repeated_sql']``) agree?"""

    name = "t2s_consistency"
    runs_key = MetaKey.REPEATED_SQL
    _rubric = _SQL_CONSISTENCY_RUBRIC
```

  4. Registration becomes `registry.register("t2s_consistency", lambda p, ctx: T2SConsistency(resolve_judge(ctx), ctx.panel))` (digest params dropped; `t2s_faithfulness` keeps them).
  5. Module docstring table row: `| ``t2s_consistency`` | no  | ``metadata['repeated_sql']`` (+ ``input``)                             |`; trim the docstring prose that describes consistency as digest-reading (faithfulness keeps that description).

- [ ] **Step 7: Run to verify pass.**

Run: `uv run pytest tests/unit/test_rag_metrics.py tests/unit/test_t2s_metrics.py tests/unit/test_registry.py tests/unit/test_runtime_critic.py`
Expected: all pass. Then `uv run ruff check src/agent_eval tests` and `uv run mypy` — clean.

---

### Task 2: Server catalog rows + server tests

**Files:**
- Modify: `src/agent_eval/server/catalog.py`
- Test: `tests/unit/test_server.py`

- [ ] **Step 1: Update the failing server tests FIRST.**

In `tests/unit/test_server.py`:

1. `HAPPY_CASES`: replace the `/rag/consistency` row with:

```python
    (
        "/rag/consistency",
        {"input": "q", "metadata": {"repeated_outputs": ["alpha beta", "alpha gamma"]}},
        0.5,  # one pair: |{alpha,beta} ∩ {alpha,gamma}| / 2
    ),
```

2. `T2S_HAPPY_CASES`: replace the `/t2s/consistency` row with:

```python
    (
        "/t2s/consistency",
        {"input": "q", "metadata": {"repeated_sql": ["SELECT name FROM emp", "SELECT id FROM emp"]}},
        0.75,  # one pair: 3 of 4 response tokens shared
    ),
```

3. REPLACE `test_consistency_accepts_repeated_runs_and_returns_final_mean` with:

```python
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
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/unit/test_server.py`
Expected: the two happy-case rows, the group test (per-item errors — old metric wants output/retrieved_context), and the digest-param test (currently 200) FAIL; the rest pass.

- [ ] **Step 3: Catalog rows.** In `src/agent_eval/server/catalog.py` replace the two consistency entries with:

```python
    EndpointSpec(
        family="rag", name="consistency", type="consistency",
        optional=("input",), metadata_keys=("repeated_outputs",), judge_based=True,
    ),
```

```python
    EndpointSpec(
        family="t2s", name="consistency", type="t2s_consistency",
        optional=("input",), metadata_keys=("repeated_sql",), judge_based=True, extra="t2s",
    ),
```

- [ ] **Step 4: Run to verify pass.**

Run: `uv run pytest tests/unit/test_server.py`
Expected: all pass (the registry cross-check test self-validates the new `requires=frozenset()`; `test_null_param_means_default`'s third case uses `/t2s/faithfulness`, unaffected). Then `uv run pytest` — whole suite green.

---

### Task 3: Predict harness `n_runs` + example configs

**Files:**
- Modify: `src/agent_eval/config/schema.py` (`PredictionModel`)
- Modify: `src/agent_eval/harness/predict.py`
- Modify: `examples/rag/rag.yaml`, `examples/t2s/t2s.yaml`
- Modify (comment-only): `examples/rag/agent.py`, `examples/t2s/agent.py`, `examples/rag/README.md`, `examples/t2s/README.md`, `examples/rag/run_eval.py`, `examples/t2s/run_eval.py`
- Test: `tests/integration/test_predict_evaluate.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/integration/test_predict_evaluate.py`):

```python
_MULTI_RUN_AGENT = '''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict, total=False):
    input: str
    output: str
    sql: str

def node(state):
    return {"output": state["input"].upper(), "sql": f"SELECT '{state['input']}'"}

_g = StateGraph(S)
_g.add_node("n", node)
_g.add_edge(START, "n")
_g.add_edge("n", END)
graph = _g.compile()
'''


def test_predict_n_runs_collects_repeated_generations(tmp_path):
    (tmp_path / "agent.py").write_text(_MULTI_RUN_AGENT)
    (tmp_path / "gold.jsonl").write_text(json.dumps({"q": "hello"}))
    cfg = ConfigModel(
        agent_type="t2s",
        datasets={
            "gold": {"adapter": "jsonl", "path": str(tmp_path / "gold.jsonl"),
                     "field_map": {"input": "q"}},
            "predictions": {"adapter": "jsonl", "path": str(tmp_path / "pred.jsonl")},
        },
        prediction={
            "graph": f"{tmp_path / 'agent.py'}:graph",
            "source": "gold", "target": "predictions", "input_key": "input",
            "state_map": {"output": "output", "sql": "sql"},
            "n_runs": 3,
        },
    )
    records = predict(cfg)
    md = records[0]["metadata"]
    # run 1 stays the canonical prediction; all 3 runs are collected as arrays
    assert records[0]["output"] == "HELLO" and md["sql"] == "SELECT 'hello'"
    assert md["repeated_outputs"] == ["HELLO", "HELLO", "HELLO"]
    assert md["repeated_sql"] == ["SELECT 'hello'", "SELECT 'hello'", "SELECT 'hello'"]


def test_predict_single_run_writes_no_repeated_arrays(tmp_path):
    records = predict(_config(tmp_path))  # n_runs defaults to 1
    assert "repeated_outputs" not in records[0]["metadata"]
    assert "repeated_sql" not in records[0]["metadata"]
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/integration/test_predict_evaluate.py`
Expected: `n_runs` test FAILS (pydantic rejects the unknown field or arrays missing); single-run test passes.

- [ ] **Step 3: Schema.** In `src/agent_eval/config/schema.py`, add to `PredictionModel` after `state_map`:

```python
    n_runs: int = Field(default=1, ge=1)  # >1 -> repeated runs for the self-consistency metrics
```

and extend the class docstring with one sentence: `"With ``n_runs > 1`` the graph runs N times per input and the repeated generations are stored as ``metadata['repeated_outputs']`` (and ``['repeated_sql']`` when ``sql`` is mapped) for the self-consistency metrics."`

- [ ] **Step 4: Harness.** In `src/agent_eval/harness/predict.py`, replace the loop body of `predict()` (lines 64-74) with:

```python
    for gold in gold_records:
        canonical = to_canonical(gold, source_spec.field_map, source_spec.include_unmapped)
        final_state, latency_ms = run_once(graph, canonical.get("input"), pred.input_key)

        projections = [to_canonical(final_state, pred.state_map, include_unmapped=False)]
        for _ in range(pred.n_runs - 1):  # repeated runs feed the self-consistency metrics
            extra_state, _ = run_once(graph, canonical.get("input"), pred.input_key)
            projections.append(to_canonical(extra_state, pred.state_map, include_unmapped=False))

        predicted = projections[0]  # run 1 stays the canonical prediction (fields + latency)
        for fld in CONTEXT_FIELDS:
            if fld in predicted:
                canonical[fld] = predicted[fld]
        canonical["metadata"].update(predicted["metadata"])
        canonical["metadata"].setdefault(MetaKey.LATENCY_MS, latency_ms)
        if pred.n_runs > 1:
            if "output" in pred.state_map:
                canonical["metadata"][MetaKey.REPEATED_OUTPUTS] = [p.get("output") for p in projections]
            if "sql" in pred.state_map:
                canonical["metadata"][MetaKey.REPEATED_SQL] = [
                    p["metadata"].get(MetaKey.SQL) for p in projections
                ]
        out.append(canonical)
```

Add one line to the module docstring workflow list: repeated runs via ``n_runs``.

- [ ] **Step 5: Run to verify pass.**

Run: `uv run pytest tests/integration/` — all pass. `uv run mypy` — clean.

- [ ] **Step 6: Example configs + stale comments.**
  - `examples/rag/rag.yaml`: after `input_key: input` insert `  n_runs: 3                               # repeated runs -> metadata['repeated_outputs'] (consistency)`
  - `examples/t2s/t2s.yaml`: after `input_key: input` insert `  n_runs: 3                               # repeated runs -> metadata['repeated_sql'] (t2s_consistency)`
  - Grep-and-fix stale semantics in example prose: `grep -rn "consistency" examples/ | grep -v predictions` — update `examples/rag/agent.py:9` (`(faithfulness / consistency)` → `(faithfulness)`), `examples/t2s/agent.py:16` (drop `t2s_consistency` from the execution_result consumers), the field tables in `examples/rag/README.md:19-20` / `examples/t2s/README.md:24-25` (consistency now reads `repeated_outputs`/`repeated_sql` produced by `n_runs`), and the suite-description comments in both `run_eval.py` files. Keep edits comment/prose-only.

- [ ] **Step 7: End-to-end example check.**

Run: `uv run agent-eval predict -c examples/rag/rag.yaml && uv run agent-eval evaluate -c examples/rag/rag.yaml`
Expected: exit 0; the report's `consistency` row shows 1.000 (deterministic agent → identical runs).
Run: `uv run python examples/t2s/run_eval.py`
Expected: exit 0; `t2s_consistency` 1.000.
The `predictions.jsonl` outputs are gitignored. If `git status --short examples/` shows *tracked* regenerated artifacts as modified (e.g. `predictions.csv` from `generate_csv: true`), restore them with `git checkout -- <file>` — the run itself was the verification; regenerated artifacts must not dirty the tree.

---

### Task 4: Docs sweep (EN + KR twins)

**Files:** `docs/metrics.md`, `docs/final_metric_list.md`, `docs/api-server.md`, `docs/langgraph-integration.md`, `docs/configuration.md`, plus grep-verified touch-ups in `docs/{concepts,judges,runtime-critic,README,repository-structure}.md` — and every edit mirrored in the `docs_kr/` twin, matching its register (declarative "~다", English terms per file convention).

- [ ] **Step 1: `docs/metrics.md`.**
  - RAG table row → `| `consistency` | no | `metadata['repeated_outputs']` (+ `input`) | mean | do repeated runs of the same query give the same answer? |`
  - The paragraph after the RAG table: replace the faithfulness/consistency sentence with: faithfulness (claims must be *supported* by the retrieved chunks) is judge-based against the evidence; `consistency` is judge-based **across repeated runs** — run the same query N times (`prediction.n_runs`) and every pair of answers is judged for agreement (score = pairwise mean; N(N−1)/2 judge calls per query).
  - T2S table row → `| `t2s_consistency` | no | `metadata['repeated_sql']` (+ `input`) | mean | do repeated runs of the same query generate equivalent SQL? |`
  - Rewrite the `## A note on the "consistency" naming` section: both metrics measure **self-consistency across repeated runs of the same query**; they differ in *what* is compared (final answers vs generated SQL), hence two registry types (`consistency`, `t2s_consistency`). Same for `faithfulness`/`t2s_faithfulness` (unchanged semantics).
- [ ] **Step 2: `docs/final_metric_list.md`.** RAG `consistency` required fields → `Repeated Final Responses (same query, N runs)`; T2S `consistency` required fields → `Repeated Generated SQL (same query, N runs)`.
- [ ] **Step 3: `docs/api-server.md`.** Endpoint-table rows: `/rag/consistency` → needs `metadata.repeated_outputs` (opt `input`), params `—`; `/t2s/consistency` → needs `metadata.repeated_sql` (opt `input`), params `—`. Rework the Request/response intro sentence: repeated runs for a consistency check ride **inside one context** (`repeated_outputs`/`repeated_sql`); posting N contexts = N separate query-groups. Note pairwise cost in the batch bullet ("consistency endpoints additionally multiply by the N(N−1)/2 run pairs").
- [ ] **Step 4: `docs/langgraph-integration.md` + `docs/configuration.md`.** Update any state-field lines tying `retrieved_context`/`output` to consistency (grep `consistency`); document `prediction.n_runs` in both files' prediction sections (configuration.md gets the field row: `n_runs` — run the graph N times per input; stores `repeated_outputs`/`repeated_sql`; default 1).
- [ ] **Step 5: Remaining EN files.** `grep -n "consistency" docs/concepts.md docs/judges.md docs/runtime-critic.md docs/README.md docs/repository-structure.md` — edit ONLY lines stating the old contradiction-vs-evidence semantics; name-level mentions stay.
- [ ] **Step 6: KR mirror.** For every EN file edited, open the `docs_kr/` twin, locate the structurally corresponding lines, and apply the equivalent edit in natural Korean matching that file's register (consistency 개념: "같은 질문을 N번 실행했을 때 답이 서로 일치하는가"; t2s: "같은 질문에 대해 생성된 SQL들이 의미적으로 동등한가"; n_runs: "같은 입력으로 그래프를 N번 실행해 반복 생성 결과를 저장한다"). Keep code/env/field names verbatim.
- [ ] **Step 7: Parity + staleness check.**

Run: `for f in metrics final_metric_list api-server langgraph-integration configuration; do echo "$f: $(grep -c '^#' docs/$f.md)/$(grep -c '^#' docs_kr/$f.md) headings, $(grep -c '^|' docs/$f.md)/$(grep -c '^|' docs_kr/$f.md) rows"; done`
Expected: equal counts per pair. Then `grep -rn "contradict" docs/ docs_kr/ --include="*.md" | grep -i consist` — expected: no hits tying consistency to contradiction-checking (faithfulness mentions may remain).

---

### Task 5: Notebooks (02_rag, 03_t2s)

**Files:** `notebook/02_rag.ipynb`, `notebook/03_t2s.ipynb` (patched via an nbformat script in the scratchpad, then re-executed headlessly).

- [ ] **Step 1: Patch cells with a script.** Write a scratchpad script that loads each notebook with `nbformat`, finds cells by matching current source substrings, and replaces them:

**02_rag.ipynb**
1. DATASET cell: each of the 3 items gains a `"repeated_outputs"` array (ASCII/numeric tokens so the stub scores instructively):
   - item 0: `["환불은 30일 이내 전액 가능합니다.", "구매 후 30일 안에 전액 환불됩니다."]` (→ 1.0)
   - item 1: `["연차는 15일입니다.", "연차는 15일이고 이월 가능합니다."]` (→ 1.0)
   - item 2: `["재택근무는 주 2회 가능합니다.", "재택근무는 주 3회 가능합니다."]` (→ 0.0)
2. SDK contexts cell: metadata gains `MetaKey.REPEATED_OUTPUTS: row["repeated_outputs"]`.
3. Section `## 1-5.` markdown → `## 1-5. \`consistency\` — 반복 실행 자기일관성\n\n같은 질문을 N번 실행했을 때 답변들이 서로 일치하는지 심판이 판단한다. 모든 쌍(pair)을 비교해 평균을 내며, 실행 횟수가 N이면 심판 호출은 N(N-1)/2번이다.\n\n> 참고: 오프라인 스텁 심판은 영숫자 토큰만 비교하므로 예시 답변에 숫자(30일, 15일 등)를 포함시켰다. 실제 LLM 심판은 한국어 문장 전체를 이해한다.` — code cell stays `run_sdk(ResponseConsistency(judge), contexts)`.
4. Section `## 2-5.` markdown → `## 2-5. \`POST /rag/consistency\`\n\n각 컨텍스트가 하나의 질문(쿼리 그룹)이고, 반복 실행 결과를 \`metadata.repeated_outputs\` 배열로 담는다.`; code cell →

```python
ctx = [{"input": r["input"], "metadata": {"repeated_outputs": r["repeated_outputs"]}} for r in DATASET]
call_api("/rag/consistency", ctx)
```

**03_t2s.ipynb**
1. DATASET cell: items gain `"repeated_sql"`:
   - item 0: `["SELECT name FROM employee WHERE salary > 100000", "SELECT name FROM employee WHERE salary > 100000"]` (→ 1.0)
   - item 1: `["SELECT dept, COUNT(*) FROM emp GROUP BY dept", "SELECT dept, COUNT(*) AS c FROM emp GROUP BY dept"]` (→ 1.0 — response tokens ⊆ reference)
   - item 2: `["SELECT name FROM product ORDER BY price DESC LIMIT 1", "SELECT MAX(price) FROM product"]` (→ 0.4)
2. SDK contexts cell: metadata gains `MetaKey.REPEATED_SQL: row["repeated_sql"]`.
3. Section `## 1-5.` markdown → `## 1-5. \`t2s_consistency\` — 반복 실행 SQL 자기일관성\n\n같은 질문을 N번 실행했을 때 생성된 SQL들이 의미적으로 동등한지 심판이 판단한다(모든 쌍 비교의 평균).`; code cell stays `run_sdk(T2SConsistency(judge), contexts)`.
4. Section `## 2-5.` markdown → `## 2-5. \`POST /t2s/consistency\`\n\n반복 생성된 SQL들을 \`metadata.repeated_sql\` 배열로 담는다. digest 관련 파라미터(\`sample_rows\` 등)는 이제 \`/t2s/faithfulness\` 전용이다.`; code cell →

```python
ctx = [{"input": r["input"], "metadata": {"repeated_sql": r["repeated_sql"]}} for r in DATASET]
call_api("/t2s/consistency", ctx)
```

Also update each notebook's intro markdown line describing consistency (02: `consistency` — 반복 실행 자기일관성; 03: `t2s_consistency` — 반복 실행 SQL 자기일관성).

- [ ] **Step 2: Re-execute headlessly.**

Run: `uv run --extra server --extra t2s --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120 --output-dir /tmp/nbval notebook/02_rag.ipynb notebook/03_t2s.ipynb`
Expected: both convert cleanly; inspect the executed copies' consistency cells — SDK and API values match ((1.0, 1.0, 0.0) for RAG; (1.0, 1.0, 0.4) for T2S). Shipped notebooks keep zero outputs. Delete `/tmp/nbval` after.

---

### Task 6: Full verification

- [ ] **Step 1:** `uv run pytest` — whole suite green (159 → expect 169: rag +5 net, t2s +2 net, server +1, integration +2).
- [ ] **Step 2:** `uv run ruff check src tests` and `uv run mypy` — clean.
- [ ] **Step 3:** Diff audit — `git status --short` modified set must be exactly: `src/agent_eval/core/contracts.py`, `src/agent_eval/metrics/{common,rag,t2s}.py`, `src/agent_eval/config/schema.py`, `src/agent_eval/harness/predict.py`, `src/agent_eval/server/catalog.py`, `tests/unit/{test_rag_metrics,test_t2s_metrics,test_server}.py`, `tests/integration/test_predict_evaluate.py`, `examples/{rag,t2s}/{rag.yaml|t2s.yaml,agent.py,README.md,run_eval.py}`, the listed `docs/` + `docs_kr/` files, plus the already-untracked `notebook/` + earlier server work. NOTHING else.
- [ ] **Step 4:** Re-run the two example pipelines once more (Task 3 Step 7 commands) as the final end-to-end proof.

---

## Self-review notes (spec → plan coverage)

- Spec §3 contracts → Task 1 Step 3. §4.1-4.3 metrics/rubrics verbatim → Task 1 Steps 4-6, pinned by Step 1 tests (pair count, exact stub values, <2-runs errors, panel confidence). §5 n_runs → Task 3 (schema+harness+integration tests+example YAMLs). §6 catalog → Task 2 (rows + 422-on-digest-params test). §7 examples → Task 3 Steps 6-7. §8 tests → Tasks 1-3 test steps. §9 docs EN+KR → Task 4 (canonical sentences + KR guidance + parity greps). §10 notebooks → Task 5 (exact cell sources + expected executed values). §11 non-changes guarded by Task 6 diff audit. §12 memory follow-up happens after the plan completes (outside repo).
- Stub-value math re-derived for every expectation in this plan: rag pairs (1.0 / 0.5 / 2⁄3 / 0.0), t2s pairs (1.0 / 0.75 / 0.4), notebook triples ((1.0,1.0,0.0), (1.0,1.0,0.4)).
