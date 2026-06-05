# Statistics & online drift

agent-eval treats every eval as a statistical experiment, not a leaderboard. This is the toolbox
(`from agent_eval.stats import ...`) and *why* each piece exists. You rarely call these directly —
the offline runner and gate use them — but they're public and composable.

## Confidence intervals (`stats.intervals`)

```python
wilson_interval(successes, n, alpha=0.05)         # -> (p, lo, hi)   DEFAULT for pass-rates
clopper_pearson(successes, n, alpha=0.05)         # exact, conservative
beta_binomial_ci(successes, n, a=1, b=1, alpha=0.05)   # Bayesian (document the prior)
percentile_ci(samples, q, alpha=0.05, n_boot=10000, rng=None)   # bootstrap CI for a quantile
```

**Why Wilson, not the textbook normal/CLT interval?** Below a few hundred datapoints the
normal-approximation interval *under-covers* (Bowyer et al., ICML 2025) — it lies about its
confidence. Per-slice golden sets in CI are usually small, so Wilson (or Clopper-Pearson) is the
default for any pass-rate. Latency is heavy-tailed → use `percentile_ci` for p95/p99, never mean±SD.

## Cluster-robust SE (`stats.clustered`)

```python
clustered_se(values, cluster_ids)                 # -> (mean, se)
```

Eval items are often **not independent**: RAG questions sharing a passage, turns in a multi-turn
scenario, tool-calls in one trajectory. Treating them as iid makes SEs far too small (Miller, 2024).
Set `EvalContext.metadata["cluster_id"]` to the unit of randomization and the offline runner uses
this automatically for graded metrics. With singleton clusters it reduces exactly to `s/√n`.

## Significance tests for regression gates (`stats.tests`)

```python
mcnemar(b, c)                                     # paired binary; b,c = discordant counts
paired_bootstrap(diffs, alpha=0.05, ...)          # paired continuous; (mean_diff, lo, hi, pvalue)
wilcoxon(a, b)                                     # paired signed-rank (non-parametric)
benjamini_hochberg(pvals, alpha=0.05)             # -> (reject_mask, qvalues)  FDR across metrics
bradley_terry(wins, n_boot=0, ...)                # rank multiple agent versions; strengths (+CIs)
```

Block a merge only when a drop is **statistically significant *and* ≥ a minimum effect size** — so
you neither ship a spurious +1% nor flag noise as a regression. `benjamini_hochberg` controls false
discoveries when you test many metrics/slices per PR. `bradley_terry` ranks N router/model variants;
declare a winner only when bootstrap CIs separate. See
[offline-evaluation.md](offline-evaluation.md#regression-gates-champion-vs-challenger).

## Judge-label validity (`stats.ppi`) & agreement (`stats.agreement`)

```python
ppi_interval(judge_labels, gold_labels, gold_idx, alpha=0.05)   # -> (theta, lo, hi)
cohen_kappa(rater_a, rater_b)                                    # chance-corrected agreement
```

`ppi_interval` (Prediction-Powered Inference) gives a **valid** CI for a judge-scored metric by
calibrating many cheap judge labels against a small human gold subset — and degrades to the
gold-only interval when the judge is noise. `cohen_kappa` is how judges are certified against humans
(see [judges.md](judges.md)).

## Reliability & economics (`stats.agents`)

```python
pass_hat_k(successes_per_task, trials_per_task, k, alpha=0.05)   # all-k-pass reliability (NOT pass@k)
cost_of_pass(success_indicators, costs_usd)                     # expected $ per correct answer
```

Production cares about **consistency**, not best-of-k — gate scenarios on **pass^k**. And fuse
accuracy with price via **Cost-of-Pass** to compare models/architectures on one Pareto-comparable
number.

## Anytime-valid sequences & online drift

`anytime_valid_sequence(stream, alpha=0.05)` returns a **time-uniform** confidence sequence — you can
check it after *every* observation without inflating the false-alarm rate (the "peeking problem").
That's what powers drift monitoring.

```python
from agent_eval.obs.drift import DriftMonitor

mon = DriftMonitor(baseline=0.85, alpha=0.05, min_n=20, sample_rate=0.1)  # watch 10% of live traffic
for trace in production_stream:
    alert = mon.observe(1.0 if trace.succeeded else 0.0)
    if alert.fired:
        rollback()   # the CS upper bound fell below the offline baseline -> a real regression
        break
```

`DriftMonitor` samples deterministically, accumulates outcomes, and **fires when the confidence
sequence's upper bound drops below the offline-established baseline** — a peeking-safe drift /
auto-rollback signal. The false-alarm rate stays ≤ `alpha` even though you check continuously.

## Emitting spans to a backend (`obs.boundary` / `obs.adapters`)

The framework emits spans through an internal boundary that fans out to swappable backends — so one
instrumentation point feeds many backends and the core never imports a vendor SDK:

```python
from agent_eval.obs.boundary import TraceBoundary, Span, span_from_node
from agent_eval.obs.adapters import MemoryBackend, JsonlBackend   # + lazy OTelBackend

boundary = TraceBoundary()
boundary.register(JsonlBackend("spans.jsonl"))       # durable
boundary.register(MemoryBackend())                   # in-process assertions/tests
boundary.emit(span_from_node("generate_sql", tokens=128, latency_ms=420.0, usd=0.002))
```

`Span` attributes use OpenTelemetry **GenAI semantic conventions** (`gen_ai.*`), so the lazy
`OTelBackend` maps cleanly to OTLP — and from there to **LangSmith / Phoenix / Langfuse** without
changing a line of framework code. Standardize at the boundary; swap backends freely.

## The principles in one list

1. Wilson/Clopper-Pearson (or Bayesian) intervals, **never CLT at small n**.
2. **Cluster** the SE on the unit of randomization for non-iid items.
3. Gate regressions on **significance + minimum effect size**, paired, with **FDR** across metrics.
4. **Certify & pin** judges; wrap judge-scored CIs in **PPI**.
5. Gate on **reliability (pass^k)** and **economics (Cost-of-Pass)**, report **p95/p99**.
6. Online: **anytime-valid** sequences so you can peek without inflating false alarms.
