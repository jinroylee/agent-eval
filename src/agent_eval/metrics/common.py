"""Common metrics — apply to **every** agent type (they score the final user-facing response).

| metric        | GT? | reads (EvalContext)                       | aggregation |
|---------------|-----|-------------------------------------------|-------------|
| ``llm_judge`` | opt | ``input``, ``output`` (+ ``expected``)    | mean        |
| ``bertscore`` | yes | ``output``, ``expected``                  | mean        |
| ``p95_latency`` | no | ``metadata['latency_ms']``               | p95 (lower=better) |
| ``token_usage`` | no | ``metadata['tokens']``                   | mean (lower=better) |

``llm_judge`` needs a configured judge backend (``defaults.judge`` in the config, or it falls back
to a deterministic token-overlap stub so examples run offline). This module also exposes
:class:`JudgeMetric`, the shared base the RAG/T2S judge metrics subclass.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetaKey, MetricResult
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import BuildContext, MetricRegistry
from agent_eval.judges.backend import (
    FunctionJudge,
    JudgeBackend,
    JudgeRequest,
    lexical_overlap_judge,
)
from agent_eval.judges.panel import poll_pointwise

DEFAULT_QUALITY_RUBRIC = (
    "Rate the overall quality of the RESPONSE TO EVALUATE as an answer to the USER QUESTION, on: "
    "Completeness (covers what was asked), Clarity (easy to understand), Usefulness (actionable, "
    "on-point), Relevance (stays on topic), and Friendliness (appropriate, helpful tone). "
    "If a REFERENCE ANSWER is given, judge how well the response matches its substance. "
    "Give a single holistic score."
)


# --------------------------------------------------------------------------- judge base
class JudgeMetric(BaseMetric):
    """Base for LLM-as-a-judge metrics: run a structured request through one judge or a PoLL panel.

    Subclasses implement :meth:`build_request` to say *what* gets graded. With a panel the metric
    averages scores and reports panel agreement as ``confidence``.
    """

    cost_class = CostClass.EXPENSIVE
    aggregation = Aggregation.MEAN

    def __init__(self, backend: JudgeBackend, panel: Sequence[JudgeBackend] = ()) -> None:
        self.backend = backend
        self.panel = list(panel)

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        raise NotImplementedError

    def _compute(self, ctx: EvalContext) -> MetricResult:
        request = self.build_request(ctx)
        judges = [self.backend, *self.panel]
        if len(judges) > 1:
            score, agreement = poll_pointwise(judges, request)
            return MetricResult(self.name, score, confidence=agreement, detail={"panel": len(judges)})
        verdict = self.backend.evaluate(request)
        return MetricResult(
            self.name, verdict.score, confidence=verdict.confidence, detail={"reason": verdict.reason}
        )


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


class LLMJudgeMetric(JudgeMetric):
    """Subjective quality of the final response (Completeness/Clarity/Usefulness/Relevance/Friendliness).

    Works with ground truth (set ``expected``) or without it. Confined to subjective quality — never
    use a judge to decide objective correctness (that's what the RAG/T2S grounded metrics are for).
    """

    name = "llm_judge"
    requires = frozenset({"input", "output"})

    def __init__(
        self,
        backend: JudgeBackend,
        panel: Sequence[JudgeBackend] = (),
        criteria: str = DEFAULT_QUALITY_RUBRIC,
        scale: tuple[int, int] = (1, 5),
    ) -> None:
        super().__init__(backend, panel)
        self.criteria = criteria
        self.scale = scale

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        return JudgeRequest(
            instruction=self.criteria,
            question=str(ctx.input),
            response=str(ctx.output),
            reference="" if ctx.expected is None else str(ctx.expected),
            scale=self.scale,
        )


# --------------------------------------------------------------------------- bertscore
class BertScore(BaseMetric):
    """Token-level semantic similarity (BERTScore F1) between the response and the gold answer.

    Catches correct answers that are worded differently (where exact match fails). Uses the
    ``bert-score`` package lazily (``[bertscore]`` extra); inject ``scorer`` to test without it.
    ``rescale_with_baseline`` makes the F1 land in a roughly ``[0, 1]`` range.
    """

    name = "bertscore"
    requires = frozenset({"output", "expected"})
    cost_class = CostClass.EXPENSIVE
    aggregation = Aggregation.MEAN

    def __init__(
        self,
        lang: str = "en",
        model_type: str | None = None,
        rescale_with_baseline: bool = True,
        scorer=None,
    ) -> None:
        self.lang = lang
        self.model_type = model_type
        self.rescale_with_baseline = rescale_with_baseline
        self._scorer = scorer

    def _get_scorer(self):
        if self._scorer is not None:
            return self._scorer
        try:
            from bert_score import score as bert_score_fn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("bertscore requires the '[bertscore]' extra (pip install bert-score)") from exc

        def scorer(cands: list[str], refs: list[str]) -> list[float]:  # pragma: no cover - needs model
            _, _, f1 = bert_score_fn(
                cands, refs, lang=self.lang, model_type=self.model_type,
                rescale_with_baseline=self.rescale_with_baseline, verbose=False,
            )
            return [float(x) for x in f1]

        return scorer

    def _compute(self, ctx: EvalContext) -> MetricResult:
        f1 = self._get_scorer()([str(ctx.output)], [str(ctx.expected)])[0]
        return MetricResult(self.name, float(f1), passed=None)


# --------------------------------------------------------------------------- performance
class P95Latency(BaseMetric):
    """Tail (95th-percentile) end-to-end latency, in milliseconds. Lower is better.

    Per item it reports ``metadata['latency_ms']`` (the prediction harness fills this automatically);
    the runner takes the p95 across the dataset. Report tails, not means — averages hide stalls.
    """

    name = "p95_latency"
    cost_class = CostClass.FREE
    aggregation = Aggregation.P95
    higher_is_better = False
    unit_interval = False

    def _compute(self, ctx: EvalContext) -> MetricResult:
        v = ctx.metadata.get(MetaKey.LATENCY_MS)
        if v is None:
            raise ValueError("metadata['latency_ms'] is required (produced by the prediction harness)")
        return MetricResult(self.name, float(v), passed=None)


class TokenUsage(BaseMetric):
    """Mean tokens spent per query. Lower is better. Reads ``metadata['tokens']``."""

    name = "token_usage"
    cost_class = CostClass.FREE
    aggregation = Aggregation.MEAN
    higher_is_better = False
    unit_interval = False

    def _compute(self, ctx: EvalContext) -> MetricResult:
        v = ctx.metadata.get(MetaKey.TOKENS)
        if v is None:
            raise ValueError("metadata['tokens'] is required (set by the harness when the run reports usage)")
        return MetricResult(self.name, float(v), passed=None)


# --------------------------------------------------------------------------- registration
def resolve_judge(ctx: BuildContext) -> JudgeBackend:
    """The configured judge, or a deterministic offline stub so examples run without an LLM.

    Shared by every judge-based metric (``llm_judge`` here, plus faithfulness/consistency in the
    RAG and T2S modules).
    """
    return ctx.judge or FunctionJudge(lexical_overlap_judge)


def register(registry: MetricRegistry) -> None:
    registry.register(
        "llm_judge",
        lambda p, ctx: LLMJudgeMetric(resolve_judge(ctx), ctx.panel, **p),
    )
    registry.register("bertscore", lambda p, ctx: BertScore(**p))
    registry.register("p95_latency", lambda p, ctx: P95Latency())
    registry.register("token_usage", lambda p, ctx: TokenUsage())
