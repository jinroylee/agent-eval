# Judge System Prompt + Per-Criteria Scores Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users customize the judge system prompt on every LLM-as-a-judge path, and make `llm_judge` report per-criterion scores (per item, in the aggregate, in reports, and on the API server).

**Architecture:** `JudgeRequest` gains a `system_prompt` field stamped by every judge metric (param > `defaults.system_prompt` > built-in default, format contract always appended). `llm_judge`'s `criteria` becomes `str | list`; a list (new default = the 5 quality criteria) issues one focused judge call per criterion through the unchanged `JudgeBackend` protocol, with per-criterion scores in `MetricResult.detail["criteria"]`, rolled up by the runner into a new optional `MetricAggregate.breakdown`.

**Tech Stack:** Python 3.12, pytest, pydantic v2, FastAPI (`[server]` extra), uv. Spec: `docs/superpowers/specs/2026-07-27-judge-system-prompt-and-criteria-breakdown-design.md`.

**Verification commands** (used throughout): `uv run pytest tests/... -q`, `uv run ruff check`, `uv run mypy`.

---

### Task 1: `system_prompt` on the judge backend

**Files:**
- Modify: `src/agent_eval/judges/backend.py`
- Test: `tests/unit/test_judges.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_judges.py`:

```python
def test_render_prompt_uses_default_system_prompt():
    assert LLMJudge.render_prompt(REQ).startswith("You are a strict, impartial evaluator.")


def test_system_prompt_precedence_request_over_backend_over_default():
    seen: list[str] = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return "SCORE: 5"

    plain = JudgeRequest(instruction="i", response="r")
    LLMJudge(complete).evaluate(plain)
    LLMJudge(complete, system_prompt="Backend persona.").evaluate(plain)
    LLMJudge(complete, system_prompt="Backend persona.").evaluate(
        JudgeRequest(instruction="i", response="r", system_prompt="Request persona.")
    )
    assert seen[0].startswith("You are a strict, impartial evaluator.")
    assert seen[1].startswith("Backend persona.")
    assert seen[2].startswith("Request persona.")


def test_custom_system_prompt_keeps_rubric_and_format_contract():
    seen: list[str] = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return "SCORE: 3"

    LLMJudge(complete).evaluate(
        JudgeRequest(instruction="rate the tone", response="r", system_prompt="Friendly persona.")
    )
    assert "rate the tone" in seen[0] and "SCORE:" in seen[0] and "REASON:" in seen[0]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_judges.py -q`
Expected: 3 failures — `TypeError: __init__() got an unexpected keyword argument 'system_prompt'` / `'system_prompt'` is an invalid `JudgeRequest` field.

- [x] **Step 3: Implement**

In `src/agent_eval/judges/backend.py`:

Add after the imports:

```python
DEFAULT_JUDGE_SYSTEM_PROMPT = "You are a strict, impartial evaluator."
```

Add a field to `JudgeRequest` (after `scale`):

```python
    system_prompt: str = ""  # persona/framing preamble; "" -> the backend's default
```

Replace `LLMJudge.__init__`, `evaluate`'s first line, and `render_prompt`'s opening:

```python
    def __init__(
        self,
        complete: Callable[[str], str],
        name: str = "llm_judge",
        system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
    ) -> None:
        self._complete = complete
        self.name = name
        self.system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        text = self._complete(self.render_prompt(request, self.system_prompt))
        ...  # rest unchanged

    @staticmethod
    def render_prompt(request: JudgeRequest, system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT) -> str:
        lo, hi = request.scale
        parts = [
            request.system_prompt or system_prompt,
            request.instruction,
            ...  # rest unchanged
```

Update the `LLMJudge` docstring: "The persona preamble is customizable (`system_prompt` here, or per request via `JudgeRequest.system_prompt` — the request wins); the rubric and the `SCORE:`/`REASON:` format lines are always appended after it, so score parsing cannot be broken."

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_judges.py -q` — Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/agent_eval/judges/backend.py tests/unit/test_judges.py
git commit -m "judge backend: customizable system prompt (request > backend > default)"
```

---

### Task 2: every judge metric stamps `system_prompt`; config plumbing

**Files:**
- Modify: `src/agent_eval/metrics/common.py` (`JudgeMetric`, `SelfConsistencyMetric`, `LLMJudgeMetric.__init__`/`build_request`, `register`, new `judge_system_prompt`)
- Modify: `src/agent_eval/metrics/rag.py` (`_ContextJudgeMetric.build_request`, `register`)
- Modify: `src/agent_eval/metrics/t2s.py` (`_ExecutionJudgeMetric`, `register`)
- Test: `tests/unit/test_common_metrics.py`, `tests/unit/test_rag_metrics.py`, `tests/unit/test_t2s_metrics.py`, `tests/unit/test_registry.py`

- [x] **Step 1: Write the failing tests**

`tests/unit/test_common_metrics.py` — add imports `pytest`, `JudgeVerdict`, and a spy at module level, plus two tests:

```python
import pytest

from agent_eval.judges.backend import FunctionJudge, JudgeVerdict, lexical_overlap_judge


class _SpyJudge:
    """Records every JudgeRequest; returns a fixed verdict."""

    def __init__(self, score: float = 1.0):
        self.requests = []
        self.score = score

    def evaluate(self, request):
        self.requests.append(request)
        return JudgeVerdict(self.score, reason="spy")


def test_llm_judge_stamps_system_prompt_on_requests():
    spy = _SpyJudge()
    LLMJudgeMetric(spy, system_prompt="Persona.").score(EvalContext(input="q", output="a"))
    assert spy.requests and all(r.system_prompt == "Persona." for r in spy.requests)


def test_judge_metric_rejects_non_string_system_prompt():
    with pytest.raises(TypeError):
        LLMJudgeMetric(FunctionJudge(lambda r: 1.0), system_prompt=42)
```

`tests/unit/test_rag_metrics.py` — add (import `JudgeVerdict` from `agent_eval.judges.backend`; copy the same `_SpyJudge` class):

```python
def test_faithfulness_stamps_system_prompt():
    spy = _SpyJudge()
    Faithfulness(spy, system_prompt="Persona.").score(
        EvalContext(input="q", output="a", retrieved_context=("c",))
    )
    assert spy.requests[0].system_prompt == "Persona."


def test_consistency_stamps_system_prompt_on_every_pair():
    spy = _SpyJudge()
    ResponseConsistency(spy, system_prompt="Persona.").score(
        EvalContext(input="q", metadata={"repeated_outputs": ["a", "b", "c"]})
    )
    assert len(spy.requests) == 3  # 3 runs -> 3 pairs
    assert all(r.system_prompt == "Persona." for r in spy.requests)
```

`tests/unit/test_t2s_metrics.py` — same pattern (copy `_SpyJudge`):

```python
def test_t2s_faithfulness_stamps_system_prompt():
    spy = _SpyJudge()
    T2SFaithfulness(spy, system_prompt="Persona.").score(
        EvalContext(input="q", output="3", metadata={MetaKey.EXECUTION_RESULT: [{"count": 3}]})
    )
    assert spy.requests[0].system_prompt == "Persona."


def test_t2s_consistency_stamps_system_prompt_on_every_pair():
    spy = _SpyJudge()
    T2SConsistency(spy, system_prompt="Persona.").score(
        EvalContext(input="q", metadata={MetaKey.REPEATED_SQL: ["SELECT 1", "SELECT 2"]})
    )
    assert spy.requests and all(r.system_prompt == "Persona." for r in spy.requests)
```

(If a test file doesn't import the named classes/`MetaKey` yet, extend its imports.)

`tests/unit/test_registry.py` — add:

```python
def test_judge_metrics_resolve_system_prompt_from_defaults_and_params():
    reg = default_registry()
    ctx = BuildContext(defaults={"system_prompt": "Global persona."})
    assert reg.build("faithfulness", ctx).system_prompt == "Global persona."
    assert reg.build("llm_judge", ctx).system_prompt == "Global persona."
    override = reg.build(
        {"type": "faithfulness", "name": "f", "params": {"system_prompt": "Metric persona."}}, ctx
    )
    assert override.system_prompt == "Metric persona."
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_common_metrics.py tests/unit/test_rag_metrics.py tests/unit/test_t2s_metrics.py tests/unit/test_registry.py -q`
Expected: new tests fail with `TypeError: ... unexpected keyword argument 'system_prompt'`.

- [x] **Step 3: Implement**

`src/agent_eval/metrics/common.py`:

`JudgeMetric.__init__` becomes:

```python
    def __init__(
        self, backend: JudgeBackend, panel: Sequence[JudgeBackend] = (), system_prompt: str = ""
    ) -> None:
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be a string; got {type(system_prompt).__name__}")
        self.backend = backend
        self.panel = list(panel)
        self.system_prompt = system_prompt
```

`SelfConsistencyMetric._compute`'s pair request gains the stamp:

```python
            request = JudgeRequest(
                instruction=self._rubric, question=str(ctx.input),
                response=runs[i], reference=runs[j],
                system_prompt=self.system_prompt,
            )
```

`LLMJudgeMetric.__init__` passes through, and `build_request` stamps:

```python
    def __init__(
        self,
        backend: JudgeBackend,
        panel: Sequence[JudgeBackend] = (),
        criteria: str = DEFAULT_QUALITY_RUBRIC,
        scale: tuple[int, int] = (1, 5),
        system_prompt: str = "",
    ) -> None:
        super().__init__(backend, panel, system_prompt=system_prompt)
        self.criteria = criteria
        self.scale = scale
```

(`build_request` adds `system_prompt=self.system_prompt` to its `JudgeRequest`; the criteria-list
rework replaces this ctor in Task 3.)

New helper next to `resolve_judge` + registration:

```python
def judge_system_prompt(p: dict, ctx: BuildContext) -> str:
    """Judge persona resolution: metric ``params`` > ``defaults.system_prompt`` > "" (built-in)."""
    return str(p.get("system_prompt") or ctx.defaults.get("system_prompt") or "")


def _build_llm_judge(p: dict, ctx: BuildContext) -> LLMJudgeMetric:
    params = {k: v for k, v in p.items() if k != "system_prompt"}
    return LLMJudgeMetric(
        resolve_judge(ctx), ctx.panel, system_prompt=judge_system_prompt(p, ctx), **params
    )


def register(registry: MetricRegistry) -> None:
    registry.register("llm_judge", _build_llm_judge)
    ...  # bertscore/p95_latency/token_usage unchanged
```

`src/agent_eval/metrics/rag.py` — import `judge_system_prompt` from `agent_eval.metrics.common`; `_ContextJudgeMetric.build_request` adds `system_prompt=self.system_prompt`; registration:

```python
    registry.register(
        "faithfulness",
        lambda p, ctx: Faithfulness(resolve_judge(ctx), ctx.panel, system_prompt=judge_system_prompt(p, ctx)),
    )
    registry.register(
        "consistency",
        lambda p, ctx: ResponseConsistency(
            resolve_judge(ctx), ctx.panel, system_prompt=judge_system_prompt(p, ctx)
        ),
    )
```

`src/agent_eval/metrics/t2s.py` — import `judge_system_prompt`; `_ExecutionJudgeMetric.__init__` gains a keyword param and passes it up; `build_request` adds the stamp:

```python
    def __init__(
        self,
        backend,
        panel=(),
        *,
        sample_rows: int = _SAMPLE_ROWS,
        max_distinct: int = _MAX_DISTINCT,
        max_columns: int = _MAX_COLUMNS,
        system_prompt: str = "",
    ) -> None:
        super().__init__(backend, panel, system_prompt=system_prompt)
        ...
```

Registration:

```python
    registry.register(
        "t2s_faithfulness",
        lambda p, ctx: T2SFaithfulness(
            resolve_judge(ctx), ctx.panel,
            system_prompt=judge_system_prompt(p, ctx), **_digest_params(p),
        ),
    )
    registry.register(
        "t2s_consistency",
        lambda p, ctx: T2SConsistency(
            resolve_judge(ctx), ctx.panel, system_prompt=judge_system_prompt(p, ctx)
        ),
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -q` — Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/agent_eval/metrics/common.py src/agent_eval/metrics/rag.py src/agent_eval/metrics/t2s.py \
  tests/unit/test_common_metrics.py tests/unit/test_rag_metrics.py tests/unit/test_t2s_metrics.py \
  tests/unit/test_registry.py
git commit -m "judge metrics: system_prompt param + defaults.system_prompt plumbing"
```

---

### Task 3: `llm_judge` per-criteria mode (new default)

**Files:**
- Modify: `src/agent_eval/metrics/common.py`
- Test: `tests/unit/test_common_metrics.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_common_metrics.py`:

```python
def test_llm_judge_default_scores_each_criterion_separately():
    spy = _SpyJudge(score=0.8)
    r = LLMJudgeMetric(spy).score(EvalContext(input="q", output="a"))
    names = ["Completeness", "Clarity", "Usefulness", "Relevance", "Friendliness"]
    assert len(spy.requests) == 5  # one focused call per default criterion
    assert [n for n in r.detail["criteria"]] == names
    assert all(name in req.instruction for name, req in zip(names, spy.requests))
    assert r.score == pytest.approx(0.8)  # mean of equal per-criterion scores
    assert set(r.detail["reasons"]) == set(names)
    assert r.error is None


def test_llm_judge_criteria_string_keeps_single_holistic_call():
    spy = _SpyJudge(score=0.6)
    r = LLMJudgeMetric(spy, criteria="Rate overall quality.").score(EvalContext(input="q", output="a"))
    assert len(spy.requests) == 1
    assert spy.requests[0].instruction == "Rate overall quality."
    assert "criteria" not in r.detail and r.detail["reason"] == "spy"


def test_llm_judge_custom_criteria_list_and_descriptions():
    spy = _SpyJudge()
    r = LLMJudgeMetric(
        spy, criteria=["Accuracy", {"name": "Tone", "description": "polite, professional"}]
    ).score(EvalContext(input="q", output="a"))
    assert len(spy.requests) == 2
    assert "Accuracy" in spy.requests[0].instruction
    assert "Tone (polite, professional)" in spy.requests[1].instruction
    assert list(r.detail["criteria"]) == ["Accuracy", "Tone"]


def test_llm_judge_criteria_panel_reports_mean_agreement():
    m = LLMJudgeMetric(FunctionJudge(lambda r: 0.8), panel=[FunctionJudge(lambda r: 0.6)])
    r = m.score(EvalContext(input="q", output="a"))
    assert r.score == pytest.approx(0.7)
    assert r.confidence == pytest.approx(0.8)  # 1 - spread, identical for every criterion
    assert r.detail["panel"] == 2
    assert all(v == pytest.approx(0.8) for v in r.detail["agreement"].values())


def test_llm_judge_per_criterion_requests_carry_shared_fields():
    spy = _SpyJudge()
    LLMJudgeMetric(spy, scale=(1, 10), system_prompt="Persona.").score(
        EvalContext(input="q", output="a", expected="ref")
    )
    for req in spy.requests:
        assert req.scale == (1, 10) and req.system_prompt == "Persona."
        assert req.question == "q" and req.response == "a" and req.reference == "ref"


@pytest.mark.parametrize(
    "bad", [[], ["ok", ""], [{"description": "no name"}], ["dup", "dup"], 42, [3]]
)
def test_llm_judge_rejects_malformed_criteria(bad):
    with pytest.raises((TypeError, ValueError)):
        LLMJudgeMetric(FunctionJudge(lambda r: 1.0), criteria=bad)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_common_metrics.py -q`
Expected: the new tests fail (5-call default and list criteria don't exist yet; malformed criteria are silently accepted).

- [x] **Step 3: Implement**

`src/agent_eval/metrics/common.py` — extend imports:

```python
from collections.abc import Mapping, Sequence
from typing import Any
```

Add after `DEFAULT_QUALITY_RUBRIC`:

```python
DEFAULT_QUALITY_CRITERIA: tuple[dict[str, str], ...] = (
    {"name": "Completeness", "description": "covers what was asked"},
    {"name": "Clarity", "description": "easy to understand"},
    {"name": "Usefulness", "description": "actionable, on-point"},
    {"name": "Relevance", "description": "stays on topic"},
    {"name": "Friendliness", "description": "appropriate, helpful tone"},
)


def _normalize_criteria(criteria: str | Sequence) -> str | tuple[dict[str, str], ...]:
    """A string is a holistic rubric (single judge call); a sequence becomes the per-criteria form."""
    if isinstance(criteria, str):
        return criteria
    if not isinstance(criteria, Sequence):
        raise TypeError(f"criteria must be a string or a list of criteria; got {type(criteria).__name__}")
    if not criteria:
        raise ValueError("criteria list must not be empty")
    normalized: list[dict[str, str]] = []
    for entry in criteria:
        if isinstance(entry, str):
            name, description = entry, ""
        elif isinstance(entry, Mapping):
            name, description = entry.get("name"), entry.get("description", "")
        else:
            raise TypeError(f"each criterion must be a string or a mapping with a 'name'; got {entry!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"criterion needs a non-empty string 'name': {entry!r}")
        if not isinstance(description, str):
            raise ValueError(f"criterion 'description' must be a string: {entry!r}")
        normalized.append({"name": name.strip(), "description": description.strip()})
    names = [c["name"] for c in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"criteria names must be unique: {names}")
    return tuple(normalized)


def _criterion_rubric(name: str, description: str) -> str:
    qualifier = f" ({description})" if description else ""
    return (
        "Rate the RESPONSE TO EVALUATE as an answer to the USER QUESTION on one criterion only: "
        f"{name}{qualifier}. If a REFERENCE ANSWER is given, judge how well the response matches "
        f"its substance on this criterion. Score only {name}; ignore all other aspects of quality."
    )
```

Replace `LLMJudgeMetric` (ctor default flips to the criteria tuple; `_compute` branches):

```python
class LLMJudgeMetric(JudgeMetric):
    """Subjective quality of the final response.

    By default every criterion in ``DEFAULT_QUALITY_CRITERIA`` is judged **separately** (one
    focused judge call each — 5 per item; a panel multiplies that): the score is the mean of the
    per-criterion scores and ``detail['criteria']`` carries the breakdown. Pass ``criteria`` as a
    plain string rubric for a single holistic call (the pre-breakdown behavior). Works with
    ground truth (set ``expected``) or without it. Confined to subjective quality — never use a
    judge to decide objective correctness (that's what the RAG/T2S grounded metrics are for).
    """

    name = "llm_judge"
    requires = frozenset({"input", "output"})

    def __init__(
        self,
        backend: JudgeBackend,
        panel: Sequence[JudgeBackend] = (),
        criteria: str | Sequence = DEFAULT_QUALITY_CRITERIA,
        scale: tuple[int, int] = (1, 5),
        system_prompt: str = "",
    ) -> None:
        super().__init__(backend, panel, system_prompt=system_prompt)
        self.criteria = _normalize_criteria(criteria)
        self.scale = scale

    def _request(self, ctx: EvalContext, instruction: str) -> JudgeRequest:
        return JudgeRequest(
            instruction=instruction,
            question=str(ctx.input),
            response=str(ctx.output),
            reference="" if ctx.expected is None else str(ctx.expected),
            scale=self.scale,
            system_prompt=self.system_prompt,
        )

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        """The holistic (string-criteria) request; list criteria go through ``_compute`` directly."""
        instruction = self.criteria if isinstance(self.criteria, str) else DEFAULT_QUALITY_RUBRIC
        return self._request(ctx, instruction)

    def _compute(self, ctx: EvalContext) -> MetricResult:
        if isinstance(self.criteria, str):
            return super()._compute(ctx)  # one holistic call; result shape unchanged
        judges = [self.backend, *self.panel]
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        agreements: dict[str, float] = {}
        for criterion in self.criteria:
            name = criterion["name"]
            request = self._request(ctx, _criterion_rubric(name, criterion["description"]))
            if len(judges) > 1:
                score, agreement = poll_pointwise(judges, request)
                agreements[name] = round(agreement, 4)
            else:
                verdict = self.backend.evaluate(request)
                score = verdict.score
                reasons[name] = verdict.reason
            scores[name] = score
        value = sum(scores.values()) / len(scores)
        detail: dict[str, Any] = {"criteria": {k: round(v, 4) for k, v in scores.items()}}
        if len(judges) > 1:
            detail["agreement"] = agreements
            detail["panel"] = len(judges)
            confidence: float | None = sum(agreements.values()) / len(agreements)
        else:
            detail["reasons"] = reasons
            confidence = None
        return MetricResult(self.name, value, confidence=confidence, detail=detail)
```

Module docstring: note the per-criteria default in the `llm_judge` row.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -q` — Expected: all pass (the pre-existing llm_judge tests use
instruction-blind judges, so their scores are unchanged by the 5-call default).

- [x] **Step 5: Commit**

```bash
git add src/agent_eval/metrics/common.py tests/unit/test_common_metrics.py
git commit -m "llm_judge: per-criteria judging as the default (one focused call per criterion)"
```

---

### Task 4: aggregate breakdown in runner + reports

**Files:**
- Modify: `src/agent_eval/core/gate.py`, `src/agent_eval/offline/runner.py`, `src/agent_eval/offline/report.py`
- Test: `tests/unit/test_runner_gate.py`, Create: `tests/unit/test_report.py`

- [x] **Step 1: Write the failing tests**

`tests/unit/test_runner_gate.py` — add `import pytest` and:

```python
class CriteriaJudge(BaseMetric):
    name = "crit"

    def _compute(self, ctx):
        crit = ctx.metadata["crit"]
        return MetricResult(self.name, sum(crit.values()) / len(crit), detail={"criteria": crit})


def test_mean_aggregate_carries_criteria_breakdown():
    data = [
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.8, "Tone": 0.6}}),
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.4, "Tone": 1.0}}),
    ]
    agg = evaluate(Suite("p", "c", [CriteriaJudge()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown == {"Clarity": pytest.approx(0.6), "Tone": pytest.approx(0.8)}


def test_breakdown_averages_only_items_carrying_each_criterion():
    data = [
        EvalContext(input="q", metadata={"crit": {"Clarity": 1.0}}),
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.0, "Tone": 0.5}}),
    ]
    agg = evaluate(Suite("p", "c", [CriteriaJudge()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown == {"Clarity": pytest.approx(0.5), "Tone": pytest.approx(0.5)}


def test_breakdown_absent_without_criteria_detail():
    data = [EvalContext(input="q", metadata={"s": 0.5})]
    agg = evaluate(Suite("p", "c", [Graded()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown is None
```

Create `tests/unit/test_report.py`:

```python
"""Reports: the per-criterion breakdown appears as text sub-rows / a JSON key only when present."""

from agent_eval.core.contracts import Aggregation
from agent_eval.core.gate import GateVerdict, MetricAggregate
from agent_eval.core.suite import SuiteResult
from agent_eval.offline import report


def _result(breakdown=None):
    agg = MetricAggregate("llm_judge", 0.7, 0.6, 0.8, 4, 0, Aggregation.MEAN, True, breakdown)
    return SuiteResult("plain", "response", 4, [agg], GateVerdict(True, {"llm_judge": True}, []))


def test_to_dict_includes_breakdown_only_when_present():
    with_bd = report.to_dict(_result({"Clarity": 0.9}))
    assert with_bd["aggregates"][0]["breakdown"] == {"Clarity": 0.9}
    assert "breakdown" not in report.to_dict(_result())["aggregates"][0]


def test_to_text_renders_criterion_sub_rows():
    text = report.to_text(_result({"Clarity": 0.9, "Tone": 0.5}))
    assert "llm_judge" in text
    assert "· Clarity" in text and "0.900" in text
    assert "· Tone" in text and "0.500" in text


def test_to_junit_is_unaffected_by_breakdown():
    assert report.to_junit([_result({"Clarity": 0.9})]) == report.to_junit([_result()])
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_runner_gate.py tests/unit/test_report.py -q`
Expected: failures — `MetricAggregate` takes no 9th positional arg / has no `breakdown`.

- [x] **Step 3: Implement**

`src/agent_eval/core/gate.py` — `MetricAggregate` gains a trailing field:

```python
    higher_is_better: bool = True
    breakdown: Mapping[str, float] | None = None  # per-criterion means (criteria-based judge metrics)
```

`src/agent_eval/offline/runner.py` — import `Mapping` from `collections.abc`; add before `_aggregate`:

```python
def _criteria_breakdown(
    valid: Sequence[tuple[MetricResult, EvalContext]],
) -> dict[str, float] | None:
    """Mean per criterion over the items whose ``detail['criteria']`` carries it (else None)."""
    per_criterion: dict[str, list[float]] = {}
    for r, _ in valid:
        crit = r.detail.get("criteria")
        if isinstance(crit, Mapping):
            for name, score in crit.items():
                per_criterion.setdefault(str(name), []).append(float(score))
    if not per_criterion:
        return None
    return {name: sum(vs) / len(vs) for name, vs in per_criterion.items()}
```

`_aggregate`'s final return becomes:

```python
    return MetricAggregate(
        metric.name, value, lo, hi, n, n_errors, agg, hib, _criteria_breakdown(valid)
    )
```

(the `n == 0` early return keeps the default `breakdown=None`.)

`src/agent_eval/offline/report.py` — in `to_dict`, replace the aggregate comprehension with a helper:

```python
def _aggregate_dict(a) -> dict:
    d = {
        "metric": a.metric,
        "value": a.value,
        "ci_low": a.ci_low,
        "ci_high": a.ci_high,
        "n": a.n,
        "n_errors": a.n_errors,
        "aggregation": a.aggregation.value,
        "higher_is_better": a.higher_is_better,
    }
    if a.breakdown:  # only when a criteria-based metric produced one — payloads otherwise unchanged
        d["breakdown"] = dict(a.breakdown)
    return d
```

with `"aggregates": [_aggregate_dict(a) for a in result.aggregates]`. In `to_text`, after each metric row:

```python
        if a.breakdown:
            lines.extend(f"    · {name:<20} {value:>7.3f}" for name, value in a.breakdown.items())
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit tests/integration -q` — Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/agent_eval/core/gate.py src/agent_eval/offline/runner.py src/agent_eval/offline/report.py \
  tests/unit/test_runner_gate.py tests/unit/test_report.py
git commit -m "aggregate per-criterion breakdown: MetricAggregate.breakdown + report rendering"
```

---

### Task 5: API server — params, response breakdown, env system prompt

**Files:**
- Modify: `src/agent_eval/server/catalog.py`, `src/agent_eval/server/schemas.py`, `src/agent_eval/server/judge.py`
- Test: `tests/unit/test_server.py`

- [x] **Step 1: Update/write the failing tests**

In `tests/unit/test_server.py`:

1. `test_catalog_declares_params_and_judges`: change the llm_judge line and append a loop:

```python
    assert by_path["/common/llm_judge"].params == ("criteria", "scale", "system_prompt")
    for s in ENDPOINTS:
        if s.judge_based:
            assert "system_prompt" in s.params, s.path
```

2. `test_result_and_aggregate_mirror_core_contracts`: build the aggregate with a breakdown and assert it:

```python
    agg = AggregateOut.from_aggregate(
        MetricAggregate("m", 0.8, 0.6, 0.9, 10, 2, Aggregation.MEAN, True, {"Clarity": 0.9})
    )
    assert (agg.metric, agg.value, agg.ci_low, agg.ci_high) == ("m", 0.8, 0.6, 0.9)
    assert (agg.n, agg.n_errors, agg.aggregation, agg.higher_is_better) == (10, 2, "mean", True)
    assert agg.breakdown == {"Clarity": 0.9}
```

3. `client` fixture: add `judge_mod.ENV_SYSTEM_PROMPT` to the cleared env vars.

4. Append the new tests:

```python
def test_llm_judge_returns_criteria_breakdown(client):
    body = client.post(
        "/common/llm_judge",
        json={"contexts": [{"input": "q", "output": "Paris", "expected": "Paris"}]},
    ).json()
    names = ["Completeness", "Clarity", "Usefulness", "Relevance", "Friendliness"]
    assert list(body["results"][0]["detail"]["criteria"]) == names
    assert body["aggregate"]["breakdown"] == {n: pytest.approx(1.0) for n in names}


def test_holistic_string_criteria_omit_breakdown(client):
    body = client.post(
        "/common/llm_judge",
        json={"contexts": [{"input": "q", "output": "Paris", "expected": "Paris"}],
              "params": {"criteria": "Rate overall quality."}},
    ).json()
    assert "criteria" not in body["results"][0]["detail"]
    assert body["aggregate"]["breakdown"] is None


def test_system_prompt_param_accepted_on_judge_endpoints(client):
    import importlib.util

    payloads = {
        "/common/llm_judge": {"input": "q", "output": "Paris", "expected": "Paris"},
        "/rag/faithfulness": {"output": "x", "retrieved_context": ["x"]},
        "/rag/consistency": {"input": "q", "metadata": {"repeated_outputs": ["a", "a"]}},
    }
    if importlib.util.find_spec("sqlglot"):
        payloads["/t2s/faithfulness"] = {
            "input": "q", "output": "3", "metadata": {"execution_result": [{"count": 3}]}
        }
        payloads["/t2s/consistency"] = {
            "input": "q", "metadata": {"repeated_sql": ["SELECT 1", "SELECT 1"]}
        }
    for path, ctx in payloads.items():
        resp = client.post(path, json={"contexts": [ctx], "params": {"system_prompt": "Persona."}})
        assert resp.status_code == 200, (path, resp.text)
        assert resp.json()["results"][0]["error"] is None


def test_malformed_criteria_and_system_prompt_are_422(client):
    for params in (
        {"criteria": 42},
        {"criteria": []},
        {"criteria": [{"description": "no name"}]},
        {"system_prompt": 42},
    ):
        resp = client.post(
            "/common/llm_judge",
            json={"contexts": [{"input": "q", "output": "a"}], "params": params},
        )
        assert resp.status_code == 422, params


def test_env_system_prompt_applies_to_openai_compatible_judge():
    import json as jsonlib

    import httpx

    from agent_eval.judges.backend import JudgeRequest
    from agent_eval.server import judge as judge_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = jsonlib.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "SCORE: 5"}}]})

    resolved = judge_mod.resolve_judge_from_env(
        environ={
            "AGENT_EVAL_JUDGE_BASE_URL": "http://llm.test/v1",
            "AGENT_EVAL_JUDGE_MODEL": "m",
            "AGENT_EVAL_JUDGE_SYSTEM_PROMPT": "Server-wide persona.",
        },
        transport=httpx.MockTransport(handler),
    )
    assert "custom system prompt" in resolved.detail
    resolved.backend.evaluate(JudgeRequest(instruction="i", response="r"))
    assert captured["body"]["messages"][0]["content"].startswith("Server-wide persona.")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_server.py -q`
Expected: failures — catalog params missing `system_prompt` (422 unknown param), `AggregateOut` has no `breakdown`, `judge_mod.ENV_SYSTEM_PROMPT` doesn't exist (fixture `AttributeError`).

- [x] **Step 3: Implement**

`src/agent_eval/server/catalog.py` — the five judge rows gain `system_prompt` in `params`:

- `/common/llm_judge`: `params=("criteria", "scale", "system_prompt")`
- `/rag/faithfulness`: `params=("system_prompt",)`
- `/rag/consistency`: `params=("system_prompt",)`
- `/t2s/faithfulness`: `params=("sample_rows", "max_distinct", "max_columns", "system_prompt")`
- `/t2s/consistency`: `params=("system_prompt",)`

`src/agent_eval/server/schemas.py` — `AggregateOut` gains:

```python
    breakdown: dict[str, float] | None = None  # per-criterion means (criteria-based judge metrics)
```

and `from_aggregate` passes `breakdown=dict(a.breakdown) if a.breakdown else None`.

`src/agent_eval/server/judge.py` — add `ENV_SYSTEM_PROMPT = "AGENT_EVAL_JUDGE_SYSTEM_PROMPT"` next to the other constants; in `resolve_judge_from_env`'s openai-compatible branch (before building `complete`):

```python
        system_prompt = env.get(ENV_SYSTEM_PROMPT, "")
        if system_prompt:
            detail_suffix += " (custom system prompt)"
```

and construct the judge as `LLMJudge(complete, name=model, system_prompt=system_prompt)`. Update the module docstring's env-var list.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_server.py -q` — Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/agent_eval/server/catalog.py src/agent_eval/server/schemas.py src/agent_eval/server/judge.py \
  tests/unit/test_server.py
git commit -m "server: system_prompt param + env default, aggregate criteria breakdown"
```

---

### Task 6: documentation (EN + KR mirrors)

**Files:**
- Modify: `docs/judges.md`, `docs/metrics.md`, `docs/configuration.md`, `docs/api-server.md`
- Modify: `docs_kr/judges.md`, `docs_kr/metrics.md`, `docs_kr/configuration.md`, `docs_kr/api-server.md`

- [x] **Step 1: `docs/judges.md`** — add `system_prompt: str = ""` to the `JudgeRequest` snippet; insert a "## The system prompt" section between "The interface" and "Wiring a real Claude judge" covering: default preamble, override levels (metric `params` / `defaults.system_prompt` / `LLMJudge(..., system_prompt=...)` / server param + `AGENT_EVAL_JUDGE_SYSTEM_PROMPT`), precedence (request > backend > built-in), the always-appended rubric + `SCORE:`/`REASON:` contract, and that `FunctionJudge`/the stub ignore it. Update the rubric table's `llm_judge` row: each criterion judged separately by default, per-criterion scores in `detail['criteria']`, plain-string `criteria` = one holistic score.

- [x] **Step 2: `docs/metrics.md`** — rewrite the `llm_judge` subsection: per-criteria default (5 focused calls/item, panel multiplies), score = mean, `detail['criteria']` + aggregate `breakdown` in reports, `criteria` accepts a list (`name` or `{name, description}`) or a plain-string rubric (single holistic call), params now `criteria`, `scale`, `system_prompt`. Mention `system_prompt` availability on the other four judge metrics in their sections' param notes.

- [x] **Step 3: `docs/configuration.md`** — `defaults:` block gains `system_prompt` with a comment (shared judge persona); the suites example shows a metric dict with `params: {criteria: [...], system_prompt: ...}`; the `defaults` row in the Sections table mentions it.

- [x] **Step 4: `docs/api-server.md`** — endpoint table `params` cells updated for the five judge rows; env table gains `AGENT_EVAL_JUDGE_SYSTEM_PROMPT`; response section notes `aggregate.breakdown`; operational notes state the llm_judge default = 5 judge calls per context (× panel).

- [x] **Step 5: Mirror all four edits in `docs_kr/`** (same sections, Korean prose matching each file's existing style).

- [x] **Step 6: Commit**

```bash
git add docs/judges.md docs/metrics.md docs/configuration.md docs/api-server.md \
  docs_kr/judges.md docs_kr/metrics.md docs_kr/configuration.md docs_kr/api-server.md
git commit -m "docs: judge system prompt + llm_judge per-criteria scores (en/kr)"
```

---

### Task 7: full verification

- [x] **Step 1: Full test suite** — Run: `uv run pytest -q` — Expected: all pass, no skips beyond the usual optional-extra skips.
- [x] **Step 2: Lint** — Run: `uv run ruff check` — Expected: `All checks passed!`
- [x] **Step 3: Types** — Run: `uv run mypy` — Expected: `Success: no issues found`.
- [x] **Step 4: Smoke the CLI report** — Run: `uv run agent-eval evaluate -c examples/plain/plain.yaml` (after `uv run agent-eval predict -c examples/plain/plain.yaml`) — Expected: exit 0; the `llm_judge` row is followed by five `·`-prefixed criterion sub-rows.
- [x] **Step 5: Commit any stragglers; do NOT include the user's unrelated `pyproject.toml` edit.**
