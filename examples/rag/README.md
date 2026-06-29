# RAG agent example

A retrieve-then-answer agent. Two suites: **retrieval** quality (does it fetch the right chunks?)
and **response** groundedness (is the answer supported by what it fetched?).

## Files

| File | What it is |
|---|---|
| [`agent.py`](agent.py) | Retrieve + answer LangGraph (`graph`) over a tiny in-file corpus |
| [`gold.jsonl`](gold.jsonl) | Questions + gold `relevant` doc ids + reference answers |
| [`rag.yaml`](rag.yaml) | Config: `defaults.k`, prediction wiring, two suites + gates |

## Required graph state fields

| canonical field | graph state key | used by |
|---|---|---|
| `retrieved_ids` | `retrieved_ids` | `recall_at_k`, `precision_at_k`, `ndcg_at_k` (ranked ids) |
| `retrieved_context` | `retrieved_context` | `faithfulness`, `consistency` (chunk texts) |
| `output` | `output` | `faithfulness`, `consistency`, `llm_judge` (the answer) |
| `tokens` | `tokens` | (available for `token_usage`) |

The gold dataset supplies `metadata['relevant_ids']` (gold ids) for the retrieval metrics.
`k` is fixed for the whole evaluation in `defaults.k` — keep it in sync with the agent's `TOP_K`.

## Run it

```bash
uv run agent-eval predict  -c examples/rag/rag.yaml
uv run agent-eval evaluate -c examples/rag/rag.yaml
```

Expected:

```
=== rag/retrieval  dataset=predictions  n=5 ===
recall_at_k                1.000        [1.000, 1.000]      0.70  PASS
precision_at_k             0.533        [0.373, 0.693]         -  info
ndcg_at_k                  1.000        [1.000, 1.000]         -  info
VERDICT: PASS

=== rag/response  dataset=predictions  n=5 ===
faithfulness               1.000        [1.000, 1.000]      0.60  PASS
consistency                1.000        [1.000, 1.000]         -  info
llm_judge                  0.773        [0.556, 0.989]         -  info
VERDICT: PASS
```

`recall_at_k` is the hard retrieval gate (it caps everything downstream); `precision_at_k` is < 1.0
because top-`k=3` includes a non-relevant doc — informational, as configured. `faithfulness` is 1.0
because the toy agent answers verbatim from the top passage; a real LLM judge would grade nuance.
