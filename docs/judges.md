# LLM-as-judge governance

Judges are the only way to score open-ended quality at scale — but the research is emphatic that
they're **biased measurement instruments**: position/verbosity/self-preference bias, overconfidence,
and near-random on objectively-checkable tasks. So agent-eval treats a judge as something you
**certify and pin** before it may gate a release, never grade objective correctness with, and whose
confidence intervals you **correct against human labels**.

## Anatomy

```python
from agent_eval.judges.config import JudgeConfig, JudgeVerdict, Protocol
from agent_eval.judges.backend import FunctionJudge, DeepEvalJudge   # pluggable backends
from agent_eval.metrics.judge_ import JudgeMetric                    # the metric (tier=JUDGE)
```

- **`JudgeConfig`** — pinned by `model | prompt_version | rubric_id` (its `fingerprint()`), plus
  `protocol` (pointwise/pairwise), `panel` (other judge ids → PoLL), `criteria`, `certification_ref`.
- **Backend** — a `JudgeBackend` has `evaluate(criteria, ctx) -> JudgeVerdict(score 0..1, confidence,
  reason)`.
  - `FunctionJudge(fn)` wraps any `fn(criteria, ctx) -> float | JudgeVerdict` (tests, rule-based).
  - `DeepEvalJudge(model, criteria)` adapts DeepEval **G-Eval** (lazy; needs `--extra deepeval` + an
    LLM). Add your own (e.g. an Azure call) by implementing the one method — see
    [customizing.md](customizing.md).
- **`JudgeMetric(backend, criteria, panel=…)`** — the scorer. With a `panel`, it averages a
  **panel-of-judges (PoLL)**; confidence = panel agreement.

```python
m = JudgeMetric(backend=FunctionJudge(lambda crit, ctx: 0.9 if "helpful" in str(ctx.output) else 0.2),
                criteria="helpfulness, correctness, instruction-following")
m.score(EvalContext("q", "a helpful answer")).score   # 0.9
```

### Panel-of-judges & pairwise

```python
from agent_eval.judges.panel import poll_pointwise, pairwise_swap_average
score, confidence = poll_pointwise([judge_a, judge_b, judge_c], criteria, ctx)  # diverse families
# pairwise with position-bias mitigation: run both orderings; flip => tie
winner = pairwise_swap_average(compare_fn, criteria, candidate_a, candidate_b)   # +1 / -1 / 0
```

Use a **diverse-family** panel (and never a judge from the same family as the agent's generator) to
cancel self-preference bias.

## Certify before you trust

A judge may gate a release only if it has a passing, fresh certification. Certify it against a small
human-labeled meta-set:

```python
from agent_eval.judges.certify import certify_judge
from agent_eval.judges.registry import JudgeRegistry

meta = [(EvalContext("q", "great answer"), 1.0), (EvalContext("q", "wrong"), 0.0), ...]  # human labels
record = certify_judge(
    judge_id="nl_intent_judge", backend=my_backend, criteria="...",
    samples=meta, fingerprint="claude-judge@2026-01|v3|t2s_nl_intent",
    kappa_floor=0.6, bias_floor=0.8,
)
record.passed       # True only if human-agreement kappa >= floor AND consistency >= floor
record.kappa, record.bias_robustness, record.n

reg = JudgeRegistry()
reg.register(judge_config)            # the JudgeConfig
reg.add_certification(record)
reg.is_certified(record.id, judge_config.fingerprint())   # True
```

`certify_judge` computes **Cohen's kappa** (chance-corrected agreement with humans) and a run-to-run
**consistency** proxy. The record is bound to the judge's **fingerprint** — change the model, prompt,
or rubric and the certification goes **stale** (`is_certified` returns `False`).

## The gate refuses uncertified judges

When a judge metric is used by a gate (it's in `gate.thresholds`/`require_pass`), suite build-out
enforces certification:

```python
from agent_eval.judges.governance import assert_gated_judges_certified
from agent_eval.core.errors import UncertifiedJudge

assert_gated_judges_certified(cfg, "response", reg)   # raises UncertifiedJudge if missing/stale/failed
```

A judge that is only **informational** (not referenced by the gate) is allowed — you can *observe*
an uncertified judge, you just can't *gate* on it. This is a hard invariant: a model-provider update
that silently shifts scores invalidates the (pinned) certification rather than corrupting your gate.

## Valid confidence intervals despite judge bias (PPI)

Naive CIs on judge scores assume the judge is ground truth. Wrap many cheap judge labels around a
small **human gold** subset with Prediction-Powered Inference so the interval stays valid:

```python
from agent_eval.judges.governance import judge_gate_ci
theta, lo, hi = judge_gate_ci(judge_scores_all, human_labels_subset, gold_idx, alpha=0.05)
```

When the judge is uninformative (noise), PPI **degrades to the gold-only interval** — so a biased or
useless judge can't manufacture false confidence.

## Full example: wire a real G-Eval judge into a gate

```python
from agent_eval.judges.backend import DeepEvalJudge
from agent_eval.judges.certify import certify_judge
from agent_eval.judges.registry import JudgeRegistry
from agent_eval.judges.config import JudgeConfig
from agent_eval.metrics.judge_ import JudgeMetric

criteria = "Does the explanation faithfully describe what the SQL computes?"
backend  = DeepEvalJudge(model="azure/gpt-4o", criteria=criteria)        # needs --extra deepeval + creds
config   = JudgeConfig(id="nl_intent", model="azure/gpt-4o", prompt_version="v1",
                       rubric_id="t2s_nl_intent", criteria=criteria, certification_ref="cert::nl_intent")

reg = JudgeRegistry(); reg.register(config)
reg.add_certification(certify_judge("nl_intent", backend, criteria, human_meta,
                                    fingerprint=config.fingerprint()))

# ... assert_gated_judges_certified(cfg, "response", reg) must pass before this metric gates:
metric = JudgeMetric(backend=backend, criteria=criteria)
```

> Tests drive judges through `FunctionJudge` stubs so they're deterministic and offline; the live
> `DeepEvalJudge` path activates only when you configure a real model + credentials.
