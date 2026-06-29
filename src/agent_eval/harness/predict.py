"""Fill a dataset with an agent's predictions by running its compiled LangGraph.

The workflow the examples follow:

1. A **gold** dataset holds the inputs + references (questions, relevant ids, gold SQL, …).
2. ``predict(cfg)`` invokes the agent's graph on each input, times it, and uses the config's
   ``state_map`` to project the resulting graph **state** onto the canonical eval fields.
3. It writes merged ``gold + prediction`` records (canonical shape) to the **target** dataset, which
   ``agent-eval evaluate`` then scores.

This is the one place the **required LangGraph state fields** are made concrete: ``state_map`` maps
canonical field -> the key your graph state exposes it under (e.g. ``output: answer``,
``sql: generated_sql``, ``retrieved_ids: doc_ids``). ``latency_ms`` is filled automatically.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agent_eval.config.loader import build_dataset_spec, import_attr
from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import MetaKey
from agent_eval.core.errors import ConfigError
from agent_eval.datasets.base import CONTEXT_FIELDS, load_records, to_canonical
from agent_eval.datasets.jsonl import write_jsonl_records


def load_graph(dotted: str) -> Any:
    """Resolve ``"module:attr"`` to a compiled LangGraph (or call it if it's a zero-arg factory)."""
    obj = import_attr(dotted)
    if hasattr(obj, "invoke"):
        return obj  # already a compiled graph
    if callable(obj):
        return obj()  # a factory returning a compiled graph
    raise ConfigError(f"{dotted!r} is neither a compiled graph (.invoke) nor a callable factory")


def run_once(graph: Any, input_value: Any, input_key: str = "input") -> tuple[dict, float]:
    """Invoke the graph with ``{input_key: input_value}``; return ``(final_state, latency_ms)``."""
    t0 = perf_counter()
    final_state = graph.invoke({input_key: input_value})
    latency_ms = (perf_counter() - t0) * 1000.0
    if not isinstance(final_state, dict):
        raise TypeError(f"graph returned {type(final_state).__name__}, expected a state dict")
    return final_state, latency_ms


def predict(cfg: ConfigModel) -> list[dict]:
    """Run the agent over the source dataset and write predictions to the target dataset.

    Returns the merged canonical records (also written to ``datasets[prediction.target].path``).
    """
    pred = cfg.prediction
    if pred is None:
        raise ConfigError("config has no 'prediction' section")

    graph = load_graph(pred.graph)
    source_spec = build_dataset_spec(cfg, pred.source)
    gold_records = load_records(source_spec)

    out: list[dict] = []
    for gold in gold_records:
        canonical = to_canonical(gold, source_spec.field_map, source_spec.include_unmapped)
        final_state, latency_ms = run_once(graph, canonical.get("input"), pred.input_key)

        predicted = to_canonical(final_state, pred.state_map, include_unmapped=False)
        for fld in CONTEXT_FIELDS:
            if fld in predicted:
                canonical[fld] = predicted[fld]
        canonical["metadata"].update(predicted["metadata"])
        canonical["metadata"].setdefault(MetaKey.LATENCY_MS, latency_ms)
        out.append(canonical)

    target_spec = build_dataset_spec(cfg, pred.target)
    if target_spec.adapter != "jsonl":
        raise ConfigError("prediction.target dataset must use the 'jsonl' adapter")
    write_jsonl_records(target_spec.path, out)
    return out
