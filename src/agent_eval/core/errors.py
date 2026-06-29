"""Typed exceptions for the framework."""

from __future__ import annotations


class AgentEvalError(Exception):
    """Base class for all framework errors."""


class ConfigError(AgentEvalError):
    """Invalid framework configuration."""
