# The runtime critic (in-flight critique)

The runtime mode answers, *mid-execution*: did this step meet its requirement, and if not, what do
we do — **accept, retry, fall back, escalate, or abstain**?

Two design rules from the research are baked in:

1. **Generator ≠ verifier.** The critic never lets the model that produced an output grade its own
   correctness on an objective task. Verification is a *separate*, grounded check.
2. **Cheap to strong.** Run a deterministic check first; only escalate to an uncertainty signal or
   (last) an LLM judge when there's no oracle and the cheap checks passed.

## The policy

```python
from agent_eval.runtime.policy import CriticPolicy
from agent_eval.core.contracts import Tier

policy = CriticPolicy(
    tiers=[Tier.DETERMINISTIC, Tier.UNCERTAINTY, Tier.JUDGE],  # escalation order
    tau={"execution_accuracy": 1.0, "selfcheck_consistency": 0.6},  # per-metric thresholds (calibrated offline)
    max_retries=2,
    fallback="abstain",                # FallbackStrategy id
    latency_budget_ms=1500,
    self_refine_only_for=["style", "format", "safety"],  # never to fix correctness
)
```

`tiers` accepts the `Tier` enum or strings (`"deterministic"`, …) — exactly what
`runtime_critic.tiers` in YAML gives you.

## The critic

A `Critic` holds metrics grouped by tier and a policy:

```python
from agent_eval.runtime.critic import Critic
from agent_eval.core.contracts import EvalContext, Decision

critic = Critic({Tier.DETERMINISTIC: [my_check]}, policy)

decision, assessment = critic.decide(EvalContext(input=..., output=...), attempt=0)
# decision: Decision.ACCEPT | RETRY | FALLBACK | ESCALATE | ABSTAIN
# assessment.accepted / .hard_fail / .confidence / .failing / .critique
```

`Critic.assess(ctx)` runs the tiers cheapest-first and stops at the first failing deterministic tier
(don't pay for expensive tiers once a grounded check fails). `Critic.decide(ctx, attempt)` maps the
assessment + attempt count to a `Decision`:

| Situation | Decision |
|---|---|
| All tiers pass | `ACCEPT` |
| Deterministic check failed, retries remain | `RETRY` |
| Deterministic check failed, retries exhausted | `FALLBACK` |
| Passed grounded checks but confidence below `tau` (no oracle) | `ESCALATE` |

`assessment.critique` is a grounded, falsifiable message built from the failing metrics' details
(e.g. *"exec_validity: no such table: emp; schema_linking: unknown tables ['emp']"*) — inject it
into your generator on retry.

## The retry→fallback loop

`critic_loop` orchestrates the whole "generate → verify → retry-with-critique → fall back" cycle —
this is what a LangGraph node ultimately calls:

```python
from agent_eval.runtime.critic import critic_loop
from agent_eval.runtime.fallback import default_fallbacks

def generate(attempt: int, critique: str | None):
    # produce an output; on retry, use `critique` to fix the previous attempt
    return my_agent(prompt, repair_hint=critique)

result = critic_loop(
    generate,
    build_ctx=lambda output: EvalContext(input=question, output=output, metadata={...}),
    critic=critic,
    fallback=default_fallbacks().get("abstain"),
)
# result.decision: ACCEPT (a good output was produced) | ABSTAIN/ESCALATE (gave up safely)
# result.output, result.attempts, result.history (per-attempt assessments)
```

**Loop guards** prevent degenerate behavior: retries are capped at `max_retries`, and if the
generator produces an output it already tried (no progress / oscillation), the loop stops and falls
back instead of spinning.

## Fallback strategies

A fallback maps `(ctx, assessment) -> (Decision, value)`. Two are built in; register your own:

```python
from agent_eval.runtime.fallback import FallbackRegistry, default_fallbacks
from agent_eval.core.contracts import Decision

reg = default_fallbacks()                 # "abstain" -> (ABSTAIN, None); "escalate" -> (ESCALATE, None)
reg.register("cached", lambda ctx, a: (Decision.FALLBACK, lookup_cache(ctx.input)))
fallback = reg.get("cached")
```

## Where do the thresholds (`tau`) come from?

From the **offline** run: `agent-eval calibrate` writes `runtime_critic.tau` so the in-flight
critic accepts/retries using thresholds the offline study justified. A single live decision (n=1)
can't do significance testing — so calibrate offline, reuse at runtime, and aggregate live outcomes
for drift monitoring (see [statistics.md](statistics.md) → drift).

## Built-in per-agent-type critics

You usually don't assemble a `Critic` by hand — each agent type ships a builder that wires the right
tiers from your config's `runtime_critic` block:

```python
from agent_eval.agents.t2s.critic import build_t2s_critic            # AST-valid + schema-linking + dry-run exec
from agent_eval.agents.rag.critic import build_rag_critic            # retrieval sufficiency (re-retrieve) + groundedness
from agent_eval.agents.plain.critic import build_plain_critic        # self-consistency hallucination flag
from agent_eval.agents.orchestration.critic import build_orchestration_critic  # per-step tool-arg validity

critic = build_t2s_critic(cfg)            # cfg from load_config(...)
```

| Agent type | Tier-1 (deterministic) | Tier-2 (uncertainty) |
|---|---|---|
| **T2S** | AST validity → schema linking → dry-run execution | — |
| **RAG** | retrieval sufficiency (Recall@k ≥ floor → re-retrieve) | semantic-entropy groundedness |
| **Plain** | — | self-consistency over sampled answers |
| **Orchestration** | per-step tool-arg validity (BFCL-style) | — |

See [langgraph-integration.md](langgraph-integration.md) for wiring any of these onto real graph
edges, and [agent-types.md](agent-types.md) for the metric details.
