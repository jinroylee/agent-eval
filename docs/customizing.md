# Customizing & extending

The framework is small on purpose. Here's how to add the things you'll most likely need.

## Add a metric

Subclass `BaseMetric` and implement `_compute`. Declare what it reads (`requires`), how it
aggregates, and its gate direction. Exceptions and missing fields are turned into
`MetricResult(error=...)` for you — you only handle the happy path.

```python
from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetricResult
from agent_eval.core.metric import BaseMetric

class AnswerLength(BaseMetric):
    name = "answer_length"
    requires = frozenset({"output"})
    cost_class = CostClass.FREE
    aggregation = Aggregation.MEAN
    higher_is_better = False        # shorter is better, say
    unit_interval = False           # raw word count, not a [0, 1] score

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, float(len(str(ctx.output).split())))
```

Use it directly (`AnswerLength().score(ctx)`), or register it so configs can name it:

```python
registry.register("answer_length", lambda params, ctx: AnswerLength(**params))
```

The factory takes `(params, ctx)` where `ctx` is a `BuildContext` carrying the resolved judge
(`ctx.judge`, `ctx.panel`) and `defaults` (`ctx.defaults["k"]`, etc.). A reference-grounded metric
reads `ctx.expected`; a metadata-based one reads `ctx.metadata['your_key']` and raises `ValueError`
if it's missing (that becomes an `error`, excluding the item).

## Add a metric via a plugin (no fork)

Advertise a factory through an entry point in your own package and `default_registry()` will pick it
up via `load_plugins()`:

```toml
# pyproject.toml of your package
[project.entry-points."agent_eval.metrics"]
answer_length = "my_pkg.metrics:make_answer_length"
```

## Add a judge backend

Implement the `JudgeBackend` protocol (one method) or reuse `LLMJudge` with any model. Point your
config's `judge.factory` at a function returning it (or a list, for a PoLL panel). See
[judges.md](judges.md).

```python
from agent_eval.judges.backend import JudgeRequest, JudgeVerdict

class MyJudge:
    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        ...  # call your model, return a normalized score
```

## Add a dataset adapter

The adapters normalize source records into `EvalContext` via a `field_map`. To support a new source,
follow `datasets/jsonl.py`: yield `record_to_context(raw, field_map, include_unmapped)` per row, and
expose a `read_*_records` helper (returning raw dicts) so the prediction harness can read inputs.
Then add a branch to `load_dataset` / `load_records` in `datasets/base.py`.

For one-off in-memory data you don't need an adapter at all — build `EvalContext` objects directly
and pass the list to `evaluate` (see [offline-evaluation.md](offline-evaluation.md)).

## Add an execution backend (T2S)

The eval metrics never touch a database — they compare **pre-computed** result sets
(`metadata['execution_result']` from the agent's state, `metadata['gold_execution_result']` from the
dataset). Execution happens earlier: the agent runs its own query at predict time, and the gold rows
are computed once at data-prep time. A read-only SQLite executor ships for that step
(`execution/harness.py` + `execution/sandbox.py`, over a `readonly_connection`); a Postgres or other
backend slots in behind the same `run(sql, db_path) -> ExecResult` interface. The result-set
comparison policy (`execution/compare.py`) is reused unchanged by `soft_f1`.

## Keep it focused

This framework deliberately ships exactly the metrics in [metrics.md](metrics.md). When you extend
it, prefer adding a metric that fills a real gap over re-deriving one that exists — and gate
objective tasks on grounded checks, not on a judge.
