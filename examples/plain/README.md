# Plain agent example

A single-shot Q&A agent — no retrieval, no tools. The **common** metrics carry the evaluation:
response quality (`llm_judge`), tail latency (`p95_latency`), and cost (`token_usage`).

## Files

| File | What it is |
|---|---|
| [`agent.py`](agent.py) | The agent under test — a tiny deterministic LangGraph (`graph`) |
| [`gold.jsonl`](gold.jsonl) | Questions + reference answers (ground truth) |
| [`plain.yaml`](plain.yaml) | Config: prediction wiring + metrics + gate |

## Required graph state fields

The config's `state_map` maps the agent's final state to the metrics' inputs:

| canonical field | graph state key | used by |
|---|---|---|
| `output` | `answer` | `llm_judge` (the response that gets graded) |
| `tokens` | `tokens` | `token_usage` |

`latency_ms` is timed automatically by the harness.

## Run it

```bash
uv run agent-eval predict  -c examples/plain/plain.yaml   # writes predictions.jsonl
uv run agent-eval evaluate -c examples/plain/plain.yaml
```

Expected (the offline stub judge; numbers vary slightly):

```
=== plain/response  dataset=predictions  n=5 ===
metric                     value                95% CI       thr  gate
llm_judge                  0.814        [0.450, 1.000]      0.50  PASS
p95_latency                  1.9        [0.300, 2.250]   2000.00  PASS
token_usage               17.400      [12.047, 22.753]         -  info
VERDICT: PASS
```

`llm_judge` is below 1.0 because one gold question (`export my data`) has no answer in the agent's
tiny knowledge base, so it falls back — exactly the kind of gap the gate is meant to catch. Swap in
a real Claude judge (see [`../judges.py`](../judges.py)) for a meaningful quality score.
