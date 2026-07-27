# Judge system prompt + per-criteria scores — design

**Date:** 2026-07-27
**Status:** approved (design review in chat); implementation pending

## 1. Goal

Two additions to every LLM-as-a-judge path (`llm_judge`, `faithfulness`, `consistency`,
`t2s_faithfulness`, `t2s_consistency`) — everything else stays the same:

1. **Customizable system prompt.** The hardcoded `"You are a strict, impartial evaluator."`
   preamble becomes the *default*, overridable per metric (YAML `params`), globally
   (`defaults.system_prompt`), per judge backend (`LLMJudge` ctor), and on the API server
   (endpoint `params` + `AGENT_EVAL_JUDGE_SYSTEM_PROMPT`). The `SCORE:`/`REASON:` format
   contract is always appended after it, so a custom prompt can never break score parsing.
2. **Per-criteria scores for `llm_judge`.** `criteria` accepts a structured list; each criterion
   is judged with its own focused request through the existing backend interface, the overall
   score is the mean, and the per-criterion scores are visible per item (`detail`), in the
   aggregate (`MetricAggregate.breakdown`), in reports (text sub-rows, JSON key), and in the
   server response.

## 2. Decisions (settled with the user)

| Question | Decision |
|---|---|
| Per-criteria mechanism | **One judge call per criterion** through the unchanged `JudgeBackend` protocol (panel applies per criterion). No multi-score parsing; custom backends participate for free. |
| Default `llm_judge` behavior | **Breakdown by default**: the default rubric becomes the structured 5-criteria list (Completeness, Clarity, Usefulness, Relevance, Friendliness) → 5 judge calls/item. A plain-string `criteria` restores today's single holistic call. |
| Aggregate visibility | Approved: per-criterion means ride on `MetricAggregate.breakdown` → CLI text sub-rows, JSON report key (present only when non-empty), server `aggregate.breakdown`. Gating stays on the overall value only. |
| System-prompt semantics | Replaces the persona/framing preamble only; per-metric rubrics (`instruction`) are unchanged and the format lines are always appended. Precedence: request > backend ctor > built-in default. |

## 3. Judge backend — `judges/backend.py`

```python
DEFAULT_JUDGE_SYSTEM_PROMPT = "You are a strict, impartial evaluator."

@dataclass(frozen=True)
class JudgeRequest:
    ...                            # existing fields unchanged
    system_prompt: str = ""        # "" → the backend's default persona/framing line
```

`LLMJudge`:

- `__init__(self, complete, name="llm_judge", system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT)`;
  stores `self.system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT`.
- `render_prompt` **stays a staticmethod** (existing call sites keep working) and gains a
  parameter: `render_prompt(request, system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT)`. The
  first prompt line becomes `request.system_prompt or system_prompt`; everything after it
  (instruction, format lines, question/evidence/reference/response blocks) is unchanged.
- `evaluate` calls `self.render_prompt(request, self.system_prompt)`.

`FunctionJudge` and the lexical stub see `request.system_prompt` but ignore it (documented).

## 4. Judge metrics — `metrics/common.py`, `rag.py`, `t2s.py`

### 4.1 `JudgeMetric` base

```python
def __init__(self, backend, panel=(), system_prompt: str = "") -> None:
    if not isinstance(system_prompt, str):
        raise TypeError(f"system_prompt must be a string; got {type(system_prompt).__name__}")
    self.backend = backend
    self.panel = list(panel)
    self.system_prompt = system_prompt
```

Every request-building site stamps `system_prompt=self.system_prompt`:
`LLMJudgeMetric` (see below), `_ContextJudgeMetric.build_request` (rag),
`_ExecutionJudgeMetric.build_request` (t2s), and the inline pair request in
`SelfConsistencyMetric._compute`.

### 4.2 `LLMJudgeMetric` — criteria as string or list

```python
DEFAULT_QUALITY_CRITERIA: tuple[dict[str, str], ...] = (
    {"name": "Completeness", "description": "covers what was asked"},
    {"name": "Clarity", "description": "easy to understand"},
    {"name": "Usefulness", "description": "actionable, on-point"},
    {"name": "Relevance", "description": "stays on topic"},
    {"name": "Friendliness", "description": "appropriate, helpful tone"},
)
```

- `DEFAULT_QUALITY_RUBRIC` (the holistic string) stays exported — it is what a plain-string
  `criteria` uses and what the single-call path sends.
- Ctor: `criteria: str | Sequence = DEFAULT_QUALITY_CRITERIA`, plus `system_prompt` passed to
  the base. `_normalize_criteria` validates a non-string sequence at build time (non-empty;
  each entry a non-empty `str` or a mapping with a non-empty string `"name"` + optional string
  `"description"`; duplicate names rejected) → `tuple[{"name", "description"}, ...]`;
  `ValueError`/`TypeError` on bad shapes (→ CLI config error / server 422 via the existing
  build guard — no server-side re-validation needed).
- Holistic path (str `criteria`): exactly today's `_compute` and `detail` shape.
- List path (`_compute` override): per criterion, build a request with
  `instruction=_criterion_rubric(name, description)` (same question/response/reference/scale/
  system_prompt), score via `backend.evaluate` or `poll_pointwise` (panel). Then:
  - score = mean of the per-criterion scores;
  - `detail = {"criteria": {name: round(score, 4)}, "reasons": {name: reason}}` (single judge)
    or `{"criteria": {...}, "agreement": {name: round(agr, 4)}, "panel": len(judges)}` (panel);
  - `confidence` = mean per-criterion agreement (panel) else `None`.

`_criterion_rubric(name, description)`: "Rate the RESPONSE TO EVALUATE as an answer to the USER
QUESTION on one criterion only — {name} ({description}). If a REFERENCE ANSWER is given, judge
how well the response matches its substance on this criterion. Score only {name}; ignore other
aspects of quality."

### 4.3 Registration — `defaults.system_prompt`

`metrics/common.py` gains (mirroring `rag._k`):

```python
def judge_system_prompt(p: dict, ctx: BuildContext) -> str:
    """system_prompt resolution: metric params > defaults.system_prompt > "" (built-in default)."""
    return str(p.get("system_prompt") or ctx.defaults.get("system_prompt") or "")
```

All five judge factories pass `system_prompt=judge_system_prompt(p, ctx)` (llm_judge forwards its
remaining params as today; rag/t2s factories otherwise unchanged, `t2s_faithfulness` keeps the
digest params). `config/schema.py` needs **no change** (`defaults` is already `dict[str, Any]`);
verify `build_suite` already threads `cfg.defaults` into `BuildContext` (it does for `k`/`dialect`).

## 5. Aggregate breakdown — `core/gate.py`, `offline/runner.py`

- `MetricAggregate` gains a trailing optional field: `breakdown: Mapping[str, float] | None = None`.
  All existing construction sites are positional-prefix compatible; `decide_gate` ignores it.
- `runner._aggregate`: after computing the aggregate value, collect `r.detail.get("criteria")`
  from valid results; if any are non-empty mappings, `breakdown = {name: mean of that
  criterion's scores over the items that carry it}` (first-seen key order). Attached for any
  aggregation kind; `None` when no result carries criteria.

## 6. Reports — `offline/report.py`

- `to_dict`: adds `"breakdown": dict(a.breakdown)` **only when present** — payloads for runs
  without criteria stay byte-identical.
- `to_text`: after a metric row with a breakdown, one indented sub-row per criterion
  (`    · {name:<20} {value:>7.3f}`).
- `to_junit`: unchanged.

## 7. Server — `catalog.py`, `schemas.py`, `judge.py` (`app.py` untouched)

- `catalog.py`: `"system_prompt"` appended to `params` of the five judge endpoints
  (`/common/llm_judge` → `("criteria", "scale", "system_prompt")`). `describe()` picks it up
  automatically.
- `schemas.py`: `AggregateOut.breakdown: dict[str, float] | None = None`; `from_aggregate`
  passes it through. (`MetricResultOut.detail` already carries the per-item criteria scores.)
- `judge.py`: `ENV_SYSTEM_PROMPT = "AGENT_EVAL_JUDGE_SYSTEM_PROMPT"` — applied as the
  `LLMJudge` ctor default on the **openai_compatible path only** (factory backends own their
  own configuration; the stub ignores it). When set, `ResolvedJudge.detail` gains a
  `" (custom system prompt)"` suffix so `/health` shows it.
- `_coerce_params` unchanged: criteria/system_prompt shape errors surface as 422 via the
  metric-ctor `ValueError`/`TypeError` through the existing build guard.

## 8. Cost note (documented, not enforced)

Default `llm_judge` now makes **5 judge calls per item** (× panel size). Stated in metrics.md,
judges.md, and api-server.md's operational notes. A plain-string `criteria` restores 1 call.

## 9. Tests (extend existing files; TDD)

- `test_judges.py` — `render_prompt`: request-level `system_prompt` wins; ctor-level next;
  built-in default when unset; format lines + rubric always present after a custom preamble.
- `test_common_metrics.py` — default llm_judge issues exactly 5 backend calls (counting
  FunctionJudge) with single-criterion instructions; `detail["criteria"]` has the 5 names;
  score = mean; string `criteria` → 1 call + today's detail shape; custom list (str + dict
  entries, description embedded in the instruction); panel → per-criterion `agreement`,
  `confidence` = mean agreement; `system_prompt` lands on every request; bad criteria
  (empty list / nameless entry / duplicate names / non-string system_prompt) raise at ctor.
  Existing single-call assumptions updated.
- `test_rag_metrics.py` / `test_t2s_metrics.py` — a spy judge asserts `request.system_prompt`
  is stamped by faithfulness + both consistency metrics (incl. every pair request).
- `test_registry.py` / `test_config.py` — `defaults.system_prompt` reaches judge metrics;
  metric `params.system_prompt` overrides it.
- `test_runner_gate.py` — breakdown: per-criterion means across items; missing-criteria items
  don't contribute; no criteria → `breakdown is None`; gate ignores breakdown.
- report tests (wherever report is covered) — text sub-rows; `to_dict` includes `breakdown`
  only when present.
- `test_server.py` — llm_judge default response carries `results[*].detail.criteria` +
  `aggregate.breakdown`; `system_prompt` accepted on all five judge endpoints; `criteria` as
  list accepted; `criteria: 42` / entry without name / `system_prompt: 42` → 422; existing
  llm_judge expectations updated (stub scores every criterion identically → same aggregate
  value, new detail shape); env `AGENT_EVAL_JUDGE_SYSTEM_PROMPT` reflected in `/health` detail.

## 10. Docs (every EN edit mirrored in `docs_kr/`, style-matched)

| File | Edit |
|---|---|
| `docs/judges.md` | New "System prompt" section (precedence chain, format-contract guarantee, env var); `JudgeRequest` snippet + rubric table updated (llm_judge → per-criterion default, panel per criterion). |
| `docs/metrics.md` | `llm_judge` section rewritten: structured default, per-criterion calls + cost, `criteria` string\|list forms, `system_prompt` param, detail/breakdown visibility. |
| `docs/configuration.md` | `defaults.system_prompt`; metric-spec example showing `criteria` list + `system_prompt`. |
| `docs/api-server.md` | Endpoint-table `params` cells; env-table row for `AGENT_EVAL_JUDGE_SYSTEM_PROMPT`; response example gains `aggregate.breakdown`; judge-call cost note (llm_judge default = 5/context). |

## 11. Non-changes

Gate semantics/statistics, `JudgeBackend` protocol, `poll_pointwise`, non-judge metrics,
datasets/harness/predict, `config/schema.py`, `server/app.py`, CLI, runtime critic, examples'
code (the stub scores criteria identically, so example gates hold), Dockerfile.

## 12. Follow-ups (out of scope unless keys available)

- `notebook/01_common.ipynb`: show the per-criteria breakdown + a custom system prompt (needs a
  real judge key to re-execute; only touched if a usable `.env` is present).
- Update the auto-memory note about judge conventions after landing.
