"""Typed exceptions for the framework."""

from __future__ import annotations


class AgentEvalError(Exception):
    """Base class for all framework errors."""


class MissingRequirement(AgentEvalError):
    """A metric was asked to score a context missing a field it requires."""


class UncertifiedJudge(AgentEvalError):
    """A gate referenced an LLM judge that has no valid certification record."""


class SelectorNotFound(AgentEvalError):
    """An EvalTarget named a node/tool/subgraph that does not exist in the graph topology."""


class CriticLoop(AgentEvalError):
    """The runtime critic detected a degenerate retry loop (oscillation / no progress)."""


class ConfigError(AgentEvalError):
    """Invalid framework configuration."""
