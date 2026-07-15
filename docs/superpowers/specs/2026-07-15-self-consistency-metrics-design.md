# Self-consistency metrics — design

**Date:** 2026-07-15
**Status:** approved (design review in chat); implementation pending

## 1. Goal

Redefine the two consistency metrics. Today `consistency` (RAG) and `t2s_consistency` judge whether
one response contradicts its evidence. From now on they measure **self-consistency across repeated
runs**: when the same user query is given N times, does the agent give the same answer (RAG) /
generate the same SQL (T2S)? Judged by the configured LLM judge. The old contradiction-vs-evidence
behavior is **replaced outright** (grounding stays covered by `faithfulness`/`t2s_faithfulness`);
the finalized 14-metric set keeps its names and size.

## 2. Decisions (settled with the user)

| Question | Decision |
|---|---|
| Runs representation | One `EvalContext` per query; all N generations in a metadata array (`repeated_outputs` / `repeated_sql`). Metric/runner/server contracts unchanged. |
| Judging | **Pairwise mean**: every unordered pair of runs is judged for semantic equivalence; score = mean over the N(N−1)/2 pairs. PoLL panel applies per pair. |
| Old behavior | Replaced outright — no renamed successor metric. |
| Harness | `prediction.n_runs: int = 1` added; >1 makes `agent-eval predict` run the graph N times per input and store the arrays. Example YAMLs use `n_runs: 3`. |
| `final_metric_list.md` | User approved editing the finalized list's consistency entries (required fields change). |

## 3. Contracts (`src/agent_eval/core/contracts.py`) — additive

```python
REPEATED_OUTPUTS = "repeated_outputs"  # consistency: the N final responses to the SAME input (one per run)
REPEATED_SQL = "repeated_sql"  # T2S consistency: the N generated SQLs for the SAME input (one per run)
```

Comment fix: `retrieved_context`'s trailing comment becomes "(for faithfulness)".

## 4. Metrics

### 4.1 Shared base — `metrics/common.py`

Module-level helper + base class (next to `JudgeMetric`, which it extends):

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
    unordered pair (i < j) is judged for semantic equivalence via
    ``JudgeRequest(instruction=rubric, question=input, response=run_i, reference=run_j)``;
    the score is the mean over pairs (panel per pair when configured).
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

Notes: `build_request` (inherited) is unused by this subclass; `itertools.combinations` import added.
Cost is N(N−1)/2 judge calls per query (× panel size) — documented in metrics.md and api-server.md.

### 4.2 `metrics/rag.py` — `ResponseConsistency`

Old `_CONSISTENCY_RUBRIC` and the `_ContextJudgeMetric`-based class are deleted
(`_ContextJudgeMetric` stays — `Faithfulness` uses it). New:

```python
_SELF_CONSISTENCY_RUBRIC = (
    "The RESPONSE TO EVALUATE and the REFERENCE ANSWER are two answers the same agent produced "
    "for the SAME user question on different runs. Judge whether they are consistent with each "
    "other: do they give the same substantive answer — the same facts, figures, and conclusion? "
    "Ignore differences in wording, order, or level of detail; penalize factual disagreement or "
    "contradictory conclusions."
)


class ResponseConsistency(SelfConsistencyMetric):
    name = "consistency"
    runs_key = MetaKey.REPEATED_OUTPUTS
    _rubric = _SELF_CONSISTENCY_RUBRIC
```

Registration unchanged: `lambda p, ctx: ResponseConsistency(resolve_judge(ctx), ctx.panel)`.
Module docstring table row: `consistency | no | metadata['repeated_outputs'] (+ input)`.

### 4.3 `metrics/t2s.py` — `T2SConsistency`

Old `_T2S_CONSISTENCY_RUBRIC` and the `_ExecutionJudgeMetric`-based class are deleted
(`_ExecutionJudgeMetric` stays — `T2SFaithfulness` uses it). New:

```python
_SQL_CONSISTENCY_RUBRIC = (
    "The RESPONSE TO EVALUATE and the REFERENCE ANSWER are two SQL queries the same agent "
    "generated for the SAME user question on different runs. Judge whether they are semantically "
    "equivalent: run against the same database, would they return the same result (same rows, "
    "columns, filters, grouping, and aggregation)? Ignore formatting, letter case, aliases, and "
    "quoting; penalize different tables, columns, filters, aggregations, or limits."
)


class T2SConsistency(SelfConsistencyMetric):
    name = "t2s_consistency"
    runs_key = MetaKey.REPEATED_SQL
    _rubric = _SQL_CONSISTENCY_RUBRIC
```

Registration **drops the digest params**: `lambda p, ctx: T2SConsistency(resolve_judge(ctx), ctx.panel)`
(`t2s_faithfulness` keeps `**_digest_params(p)`). Module docstring table row:
`t2s_consistency | no | metadata['repeated_sql'] (+ input)`.

## 5. Predict harness — `n_runs`

`config/schema.py` `PredictionModel`: `n_runs: int = Field(default=1, ge=1)` — ">1 → run the graph
N times per input and store the repeated generations for the self-consistency metrics."

`harness/predict.py`: run 1 fills the canonical record exactly as today (output/sql/latency from
run 1). When `n_runs > 1`, the remaining runs are invoked and projected through the same
`state_map`; then, convention-based:

- `metadata['repeated_outputs']` = each run's mapped `output` — written iff `"output" in pred.state_map`.
- `metadata['repeated_sql']` = each run's mapped `sql` — written iff `"sql" in pred.state_map`.

Extra runs' latencies are not recorded (`latency_ms` stays run 1's, as today).

## 6. Server — `server/catalog.py` rows only (handler/schemas/judge untouched)

```python
EndpointSpec(
    family="rag", name="consistency", type="consistency",
    optional=("input",), metadata_keys=("repeated_outputs",), judge_based=True,
),
EndpointSpec(
    family="t2s", name="consistency", type="t2s_consistency",
    optional=("input",), metadata_keys=("repeated_sql",), judge_based=True, extra="t2s",
),
```

Consequences: `/t2s/consistency` no longer accepts `sample_rows`/`max_distinct`/`max_columns`
(unknown params → 422); `requires` stays `()` matching the metric's `requires = frozenset()`.
API narrative: post M contexts = M query-groups, each carrying its own runs array; the aggregate is
the mean consistency across queries.

## 7. Examples

- `examples/rag/rag.yaml` and `examples/t2s/t2s.yaml`: add `n_runs: 3` to `prediction`. Suites keep
  their consistency entries (the deterministic example agents give identical runs → 1.0).
- Comment/README lines stating the old semantics updated:
  `examples/rag/agent.py` and `examples/t2s/agent.py` header comments, `examples/rag/README.md` /
  `examples/t2s/README.md` field tables, `examples/*/run_eval.py` suite-description comments.

## 8. Tests

- `tests/unit/test_rag_metrics.py` / `test_t2s_metrics.py` — old consistency tests replaced:
  identical runs → 1.0; a counting judge proves 3 runs ⇒ exactly 3 pair calls; a divergent pair
  scores the exact stub value; <2 runs / missing key / non-list → `error` set (not 0.0);
  panel → `confidence` = mean pair agreement; `detail` carries `n_runs` + `pair_scores`.
- `tests/unit/test_server.py` — the two consistency `HAPPY_CASES`/`T2S_HAPPY_CASES` rows get new
  payloads (`metadata.repeated_outputs` / `metadata.repeated_sql`) with exact stub expectations;
  `test_consistency_accepts_repeated_runs_and_returns_final_mean` is rewritten to post M=3
  query-groups (mixed consistency) and assert the mean; a new test asserts `sample_rows` on
  `/t2s/consistency` is now a 422 (unknown param).
- Everything else must stay green (`uv run pytest`, ruff, mypy).

**Stub-judge note (tests & notebooks):** the offline stub tokenizes `[a-z0-9]+` only — Korean-only
strings produce empty token sets (score 0). Demo/test run-arrays therefore carry ASCII/numeric
tokens (SQL is naturally ASCII; Korean demo answers include figures like "30일"). Real LLM judges
handle Korean text; the notebooks say so explicitly.

## 9. Docs (every EN edit mirrored in `docs_kr/`, style-matched)

| File | Edit |
|---|---|
| `docs/metrics.md` | RAG + T2S table rows (reads/one-liner), the surrounding prose, and the "note on the consistency naming" section rewritten around self-consistency; pairwise cost note. |
| `docs/final_metric_list.md` | consistency (RAG): required fields → Repeated Final Responses (same query); consistency (T2S): → Repeated Generated SQL (same query). |
| `docs/api-server.md` | Two endpoint-table rows; the "repeated runs" batch narrative sentence reworded (runs now live inside one context). |
| `docs/langgraph-integration.md` | State-field lines mapping `retrieved_context`/`output` to consistency updated; mention `n_runs`. |
| `docs/configuration.md` | `prediction.n_runs` documented. |
| `docs/concepts.md`, `docs/judges.md`, `docs/runtime-critic.md`, `docs/README.md`, `docs/repository-structure.md` | Touch only lines that state the old semantics (grep-verified); name-level mentions stay. |

## 10. Notebooks

`notebook/02_rag.ipynb` / `notebook/03_t2s.ipynb`: the consistency sub-sections (SDK + API) get new
demo data — each dataset item gains `repeated_outputs` / `repeated_sql` (one clearly consistent
item, one divergent), SDK cells call the new metrics, API cells post the arrays; Korean prose
updated (including the stub-tokenizer note). Both notebooks re-executed headlessly to validate.

## 11. Non-changes

Metric names/registry types, MEAN aggregation, EXPENSIVE cost class, gate/runner/stats,
`faithfulness`/`t2s_faithfulness` (incl. digest params), server `app.py`/`schemas.py`/`judge.py`,
Dockerfile, `01_common.ipynb`. `runtime/critic.py` code is untouched (its docstring's
"consistency → escalate" example remains valid: still a graded judge metric).

## 12. Follow-up (out of scope)

Update the auto-memory note about the finalized metric set after implementation lands.
