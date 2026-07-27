"""Resolve the server's LLM judge from environment variables (startup-time, server-wide).

Precedence:

1. ``AGENT_EVAL_JUDGE_FACTORY`` — the exact YAML ``judge.factory`` convention (``module:attr`` or
   ``path/to/file.py:attr``; may return a backend, a zero-arg callable, or a list = PoLL panel).
   Resolved through :func:`agent_eval.config.loader.resolve_judge`, so semantics never drift.
2. ``AGENT_EVAL_JUDGE_BASE_URL`` + ``AGENT_EVAL_JUDGE_MODEL`` (+ optional ``_API_KEY``,
   ``_TIMEOUT`` seconds, ``_SYSTEM_PROMPT`` — the server-wide judge persona; a request's
   ``system_prompt`` param overrides it) — any OpenAI-compatible ``/chat/completions`` (vLLM, an
   internal gateway). The framework's own :class:`LLMJudge` renders the prompt and parses
   ``SCORE:``; this module contributes only the ``complete(prompt) -> str`` callable.
3. ``AGENT_EVAL_JUDGE_PROVIDER`` — a preset switch used when no explicit ``_BASE_URL`` is given:
   ``anthropic`` (Claude via Anthropic's OpenAI-compatible endpoint, needs ``ANTHROPIC_API_KEY``)
   or ``openai`` (needs ``OPENAI_API_KEY``); ``_MODEL`` overrides the preset's default model.
4. None of the above — the deterministic lexical-overlap stub (offline; NOT a real judgment —
   ``/health`` reports the active kind so stub scores are never mistaken for real ones).

The server never reads a dotenv file implicitly; ``agent-eval-serve`` wires that up explicitly
via :func:`load_env_file` (``--env-file``, default ``./.env``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_eval.config.loader import resolve_judge
from agent_eval.config.schema import ConfigModel, JudgeModel
from agent_eval.judges.backend import FunctionJudge, JudgeBackend, LLMJudge, lexical_overlap_judge

ENV_FACTORY = "AGENT_EVAL_JUDGE_FACTORY"
ENV_BASE_URL = "AGENT_EVAL_JUDGE_BASE_URL"
ENV_MODEL = "AGENT_EVAL_JUDGE_MODEL"
ENV_API_KEY = "AGENT_EVAL_JUDGE_API_KEY"
ENV_TIMEOUT = "AGENT_EVAL_JUDGE_TIMEOUT"
ENV_PROVIDER = "AGENT_EVAL_JUDGE_PROVIDER"
ENV_SYSTEM_PROMPT = "AGENT_EVAL_JUDGE_SYSTEM_PROMPT"

# Preset endpoints the provider switch maps to (Anthropic via its OpenAI-compatible surface).
_PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    # provider → (base_url, default judge model, provider-specific key env var)
    "anthropic": ("https://api.anthropic.com/v1", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
}


def load_env_file(path: str | Path) -> bool:
    """Load ``KEY=VALUE`` lines from a dotenv-style file into ``os.environ``.

    Setdefault semantics — already-set variables always win; comment/blank lines and lines
    without ``=`` are skipped; values are taken verbatim (no quoting dialect). Returns True when
    the file existed and was read. Kept deliberately explicit: the server only loads a file when
    a caller asks (``agent-eval-serve --env-file``, or a notebook), never at import time.
    """
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    return True


@dataclass(frozen=True)
class ResolvedJudge:
    """The judge the server will inject into every metric build, plus how it was chosen."""

    backend: JudgeBackend
    panel: tuple[JudgeBackend, ...] = ()
    kind: str = "stub"  # stub | factory | openai_compatible | injected
    detail: str = ""


def openai_complete(
    base_url: str, model: str, api_key: str | None, timeout: float, transport: Any | None = None
) -> Callable[[str], str]:
    """A ``complete(prompt) -> str`` callable against an OpenAI-compatible endpoint.

    ``transport`` is a test hook (``httpx.MockTransport``); leave ``None`` in production.
    """
    import httpx  # deferred: only this judge path needs it

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # One shared, thread-safe client reused across calls (a judge scores many items per request);
    # it lives for the process lifetime, like the server's judge itself.
    client = httpx.Client(timeout=timeout, transport=transport)

    def complete(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return complete


def resolve_judge_from_env(
    environ: Mapping[str, str] | None = None, transport: Any | None = None
) -> ResolvedJudge:
    """Resolve the judge per the documented precedence. ``environ`` overrides ``os.environ`` (tests)."""
    env = os.environ if environ is None else environ

    factory = env.get(ENV_FACTORY)
    if factory:
        backend, panel = resolve_judge(ConfigModel(judge=JudgeModel(factory=factory)))
        if backend is None:
            raise RuntimeError(f"{ENV_FACTORY}={factory!r} did not produce a judge backend")
        return ResolvedJudge(backend, tuple(panel), kind="factory", detail=factory)

    base_url, model = env.get(ENV_BASE_URL), env.get(ENV_MODEL)
    api_key = env.get(ENV_API_KEY)
    detail_suffix = ""
    provider = env.get(ENV_PROVIDER, "").strip().lower()
    if not base_url and provider:  # the preset fills the endpoint; an explicit _BASE_URL wins
        if provider not in _PROVIDER_PRESETS:
            raise RuntimeError(
                f"{ENV_PROVIDER}={provider!r} is not a known provider; "
                f"choose one of {sorted(_PROVIDER_PRESETS)} or set {ENV_BASE_URL} directly"
            )
        base_url, preset_model, key_var = _PROVIDER_PRESETS[provider]
        model = model or preset_model
        api_key = api_key or env.get(key_var)
        if not api_key:
            raise RuntimeError(f"{ENV_PROVIDER}={provider} needs {key_var} (or {ENV_API_KEY}) set")
        detail_suffix = f" (provider={provider})"

    if base_url or model:
        if not base_url:
            raise RuntimeError(f"{ENV_MODEL} is set but {ENV_BASE_URL} is missing")
        if not model:
            raise RuntimeError(f"{ENV_BASE_URL} is set but {ENV_MODEL} is missing")
        raw_timeout = env.get(ENV_TIMEOUT, "60")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise RuntimeError(f"{ENV_TIMEOUT}={raw_timeout!r} is not a number") from exc
        system_prompt = env.get(ENV_SYSTEM_PROMPT, "")
        if system_prompt:  # surfaced in /health so a custom persona is never invisible
            detail_suffix += " (custom system prompt)"
        complete = openai_complete(base_url, model, api_key, timeout, transport)
        return ResolvedJudge(
            LLMJudge(complete, name=model, system_prompt=system_prompt),
            kind="openai_compatible",
            detail=f"{model} @ {base_url}{detail_suffix}",
        )

    return ResolvedJudge(
        FunctionJudge(lexical_overlap_judge), kind="stub", detail="lexical_overlap_judge"
    )
