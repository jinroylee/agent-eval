"""LLM-as-a-judge backends — pluggable scorers behind one small interface.

A metric builds a structured :class:`JudgeRequest` (instruction + the fields to grade); a backend
turns it into a :class:`JudgeVerdict` (score in ``[0, 1]`` + reason). Three backends ship:

- :class:`LLMJudge`   — provider-agnostic: you pass a ``complete(prompt) -> text`` callable, so any
  model (Claude, GPT, a local model) works without this package depending on any SDK.
- :class:`FunctionJudge` — wraps ``fn(request) -> score`` for deterministic/offline judging + tests.
- :func:`lexical_overlap_judge` — a ready-made deterministic stub (token-overlap) so the bundled
  examples run with no API key; swap in :class:`LLMJudge` for real evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

DEFAULT_JUDGE_SYSTEM_PROMPT = "You are a strict, impartial evaluator."


@dataclass(frozen=True)
class JudgeRequest:
    """A self-contained grading task. A metric fills the fields relevant to what it measures."""

    instruction: str  # the rubric: what to grade and how
    question: str = ""  # the user input, if relevant
    response: str = ""  # the response under test (graded)
    reference: str = ""  # gold answer, if grading against ground truth
    context: Sequence[str] = field(default_factory=tuple)  # evidence (chunks, execution rows)
    scale: tuple[int, int] = (1, 5)  # integer score range the judge is asked to use
    system_prompt: str = ""  # persona/framing preamble; "" -> the backend's default


@dataclass(frozen=True)
class JudgeVerdict:
    score: float  # normalized to [0, 1]
    confidence: float | None = None
    reason: str = ""


@runtime_checkable
class JudgeBackend(Protocol):
    def evaluate(self, request: JudgeRequest) -> JudgeVerdict: ...


class FunctionJudge:
    """Wrap ``fn(request) -> float in [0, 1]`` (or a JudgeVerdict) as a JudgeBackend."""

    def __init__(self, fn: Callable[[JudgeRequest], float | JudgeVerdict]) -> None:
        self._fn = fn

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        out = self._fn(request)
        return out if isinstance(out, JudgeVerdict) else JudgeVerdict(float(out))


class LLMJudge:
    """Score with any LLM via a ``complete(prompt) -> completion_text`` callable.

    The model is asked to answer with ``SCORE: <int>`` (in the request's scale) and ``REASON: ...``;
    the score is parsed and normalized to ``[0, 1]``. Provider-agnostic on purpose — wire Claude,
    GPT, or a local model by supplying the callable (see docs/judges.md).

    The persona preamble is customizable: ``system_prompt`` here sets the backend-wide default and
    ``JudgeRequest.system_prompt`` overrides it per request (the request wins). The rubric and the
    ``SCORE:``/``REASON:`` format lines are always appended after it, so a custom system prompt can
    never break score parsing.
    """

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
        score01, raw = self._parse(text, request.scale)
        reason = ""
        m = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            reason = m.group(1).strip()
        return JudgeVerdict(score01, reason=reason, confidence=None)

    @staticmethod
    def render_prompt(request: JudgeRequest, system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT) -> str:
        lo, hi = request.scale
        parts = [
            request.system_prompt or system_prompt,
            request.instruction,
            f"\nReturn your verdict as two lines exactly:"
            f"\nSCORE: <integer {lo}-{hi}>\nREASON: <one sentence>",
        ]
        if request.question:
            parts.append(f"\n[USER QUESTION]\n{request.question}")
        if request.context:
            joined = "\n".join(f"- {c}" for c in request.context)
            parts.append(f"\n[EVIDENCE]\n{joined}")
        if request.reference:
            parts.append(f"\n[REFERENCE ANSWER]\n{request.reference}")
        parts.append(f"\n[RESPONSE TO EVALUATE]\n{request.response}")
        return "\n".join(parts)

    @staticmethod
    def _parse(text: str, scale: tuple[int, int]) -> tuple[float, int]:
        lo, hi = scale
        m = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if not m:
            raise ValueError(f"judge response had no parseable SCORE: {text!r}")
        raw = float(m.group(1))
        raw = max(lo, min(hi, raw))
        return (raw - lo) / (hi - lo) if hi > lo else 1.0, int(raw)


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def lexical_overlap_judge(request: JudgeRequest) -> JudgeVerdict:
    """Deterministic offline stand-in: how well is the response covered by the evidence/reference?

    Token-recall of the response against (context + reference). Not a real judge — it just lets the
    examples run end-to-end without an LLM. Replace with :class:`LLMJudge` for meaningful scores.
    """
    resp = _tokens(request.response)
    if not resp:
        return JudgeVerdict(0.0, reason="empty response")
    evidence = set().union(*[_tokens(c) for c in request.context]) if request.context else set()
    evidence |= _tokens(request.reference)
    if not evidence:
        return JudgeVerdict(0.5, reason="no evidence to compare against")
    covered = len(resp & evidence) / len(resp)
    return JudgeVerdict(covered, reason=f"{covered:.0%} of response tokens supported by evidence")
