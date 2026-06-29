# Offline evaluation (the CI/CD gate)

Offline mode answers: *is this agent good enough to deploy?* — with statistics, not vibes.

## The flow

1. You have a **gold dataset** of labeled examples (references / relevant ids / gold queries).
2. **`agent-eval predict`** runs your agent over the inputs and writes a **predictions** dataset
   (skip this if you already have predictions — see [langgraph-integration.md](langgraph-integration.md)).
3. A **suite** of metrics scores each prediction.
4. The runner **aggregates** per metric with small-sample-correct confidence intervals.
5. A **gate** turns that into a pass/fail verdict; the CLI exits non-zero on failure.

## Run it

```bash
uv run agent-eval evaluate --config eval.yaml
# --category <name>   run one suite only (default: all)
# --dataset <name>    override which configured dataset to score
```

```
=== rag/retrieval  dataset=predictions  n=5 ===
metric                     value                95% CI       thr  gate
recall_at_k                0.900        [0.704, 1.000]      0.70  PASS
precision_at_k             0.633        [0.417, 0.850]         -  info
VERDICT: PASS
```

## From Python

```python
from agent_eval.core.contracts import EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.metrics.rag import RecallAtK
from agent_eval.offline.runner import evaluate

dataset = [
    EvalContext(input="q1", metadata={"retrieved_ids": ["d1", "d2"], "relevant_ids": ["d1"]}),
    EvalContext(input="q2", metadata={"retrieved_ids": ["x"], "relevant_ids": ["d3"]}),
]
suite = Suite("rag", "retrieval", [RecallAtK(k=10)],
              GatePolicy(thresholds={"recall_at_k": 0.4}, require_pass=["recall_at_k"]))
result = evaluate(suite, dataset)

print(result.verdict.passed)
for agg in result.aggregates:
    print(agg.metric, agg.value, (agg.ci_low, agg.ci_high), f"n={agg.n}", f"errors={agg.n_errors}")
```

`evaluate(suite, dataset, alpha=0.05)` returns a `SuiteResult(agent_type, category, n_items,
aggregates, verdict)` where each `MetricAggregate` carries `value`, `ci_low/ci_high`, `n`,
`n_errors`, `aggregation`, and `higher_is_better`; `GateVerdict` carries `passed`, `metric_passed`,
and `reasons`.

## How aggregation works (and why)

Each metric declares an `aggregation`; the runner uses small-sample-correct statistics for each:

- **`RATE`** (binary metrics like `ast_valid`) → pass rate with a **Wilson** interval, which doesn't
  under-cover at small *n* the way the normal-approximation/CLT interval does.
- **`MEAN`** (graded metrics like `recall_at_k`, `soft_f1`, `faithfulness`, a judge score) → mean
  with a **cluster-robust** SE. If your items aren't independent — RAG questions sharing a passage,
  multi-turn scenarios — set `metadata["cluster_id"]` so the error bar isn't artificially tight.
- **`P95`** (`p95_latency`) → 95th percentile with a **bootstrap** interval (latency is heavy-tailed,
  so a normal approximation would be wrong).

**Errors** (a metric that couldn't run — missing field, exception, bad gold) are counted in
`n_errors` and excluded from `n`; they surface in the gate reasons. A genuine bad result just scores
low — the two are never conflated.

## The gate

Every metric with a configured threshold must meet it, **in its own direction**: `recall_at_k: 0.7`
means ≥ 0.7, `p95_latency: 2000` means ≤ 2000. `require_pass` metrics are hard blockers that must
have run and passed. Metrics without a threshold are informational.

## Reporting (JSON / JUnit / text)

```python
from agent_eval.offline import report
report.to_json(result)     # durable machine artifact
report.to_junit([result])  # JUnit XML for CI dashboards; failing thresholds become <failure>
report.to_text(result)     # the human table the CLI prints
```

## CI integration

```yaml
# .github/workflows/eval.yml (sketch)
- run: uv sync --extra langgraph --extra t2s
- run: uv run agent-eval predict  -c eval/rag.yaml
- run: uv run agent-eval evaluate -c eval/rag.yaml   # non-zero exit fails the job
```

Keep a **frozen** golden slice in version control for apples-to-apples comparisons across releases;
add a rolling supplement from production failures for realism. Version every dataset and link it to
the agent release it gated.
