# Runtime critique (foundation)

> **Status: a foundation for a future mode, not part of the offline gate.** The offline
> `predict`/`evaluate` path does not import any of this. It's kept here, wired against the current
> metric contract and covered by tests, so the runtime-critique mode can be built on top of it later.

The offline runner scores a whole *dataset* and gates a release. The **critic** is the in-flight
counterpart: it scores a *single step* with the **same `Metric` objects** and turns the scores into a
decision, so a LangGraph node can self-correct before emitting a bad result. One metric core, two
entrypoints.

## Design principles (preserved for the build-out)

- **Cheap-first escalation.** Metrics run in `cost_class` order (FREE → CHEAP → EXPENSIVE) and the
  critic short-circuits the moment a grounded check fails — so you never pay for the LLM judge when a
  free parse/execution check already caught the problem.
- **Generator/verifier separation.** The critic consumes only *external* verifiers (`ast_valid`,
  retrieval, an independent judge); it never asks the model that produced an answer to grade its own
  correctness.
- **Hard vs soft failure.** A failed binary/grounded check (`aggregation == RATE`, e.g. `ast_valid`)
  is an objective error → **retry**. A low-confidence graded check (`consistency`, `faithfulness`) →
  **escalate** for review. Everything passing → **accept**.

## The pieces (`agent_eval.runtime`)

| | |
|---|---|
| `Critic(metrics, policy)` | `assess(ctx) -> Assessment`; `decide(ctx, attempt) -> (Decision, Assessment)` |
| `CriticPolicy(tau, max_retries)` | per-metric acceptance thresholds + retry budget |
| `Decision` | `ACCEPT · RETRY · FALLBACK · ESCALATE · ABSTAIN` |
| `critic_loop(generate, build_ctx, critic, fallback)` | bounded generate → verify → retry-with-critique loop |
| `LoopGuard` | stops oscillation / no-progress retries |
| `default_fallbacks()` | named strategies (`abstain`, `escalate`) |

## Sketch

The critic is built from exactly the metrics you already use — e.g. a T2S step gated on `ast_valid`
(grounded, hard) with a `t2s_faithfulness` confidence check (graded, soft):

```python
from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.metrics.t2s import AstValid, T2SFaithfulness
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.runtime.critic import Critic, CriticPolicy, critic_loop
from agent_eval.runtime.fallback import default_fallbacks

critic = Critic(
    metrics=[AstValid(), T2SFaithfulness(FunctionJudge(lexical_overlap_judge))],
    policy=CriticPolicy(tau={"t2s_faithfulness": 0.6}, max_retries=2),
)

def generate(attempt, critique):
    # call your model; inject `critique` (the grounded feedback) on retries
    ...

def build_ctx(sql_and_answer):
    return EvalContext(input=question, output=sql_and_answer.answer,
                       metadata={MetaKey.SQL: sql_and_answer.sql, MetaKey.DB_REF: db})

result = critic_loop(generate, build_ctx, critic, fallback=default_fallbacks().get("abstain"))
result.decision   # ACCEPT / ESCALATE / ABSTAIN / ...
```

## What's intentionally left for later

- No `agent-eval` CLI command and no config section (the offline flow stays the simple surface).
- The `tau` thresholds are hand-set; the planned design calibrates them from an offline study so the
  in-flight critic and the CI gate share operating points.
- No LangGraph node wrapper yet — `critic_loop` is framework-agnostic; wiring it into a node
  (inject the critique on retry, route on the `Decision`) is the next step.
