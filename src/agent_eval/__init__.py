"""agent-eval — an offline evaluation framework for LangGraph agents.

Convenience re-exports for the common path:

    from agent_eval import EvalContext, Suite, GatePolicy, evaluate
"""

from agent_eval.core.contracts import Aggregation, EvalContext, MetaKey, MetricResult
from agent_eval.core.gate import GatePolicy
from agent_eval.core.registry import default_registry
from agent_eval.core.suite import Suite, SuiteResult
from agent_eval.offline.runner import evaluate

__all__ = [
    "EvalContext",
    "MetricResult",
    "MetaKey",
    "Aggregation",
    "Suite",
    "SuiteResult",
    "GatePolicy",
    "evaluate",
    "default_registry",
]
