"""Online drift monitoring with an anytime-valid confidence sequence.

Watches a metric (e.g. per-trace success) on sampled live traffic and fires when the confidence
sequence's upper bound falls below the offline-established baseline — i.e. the metric has
significantly regressed. Because the sequence is time-uniform, you can check after every trace
without inflating the false-alarm rate (the peeking problem) — suitable for drift alerts and
auto-rollback.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.stats.sequential import anytime_valid_sequence


@dataclass
class DriftAlert:
    fired: bool
    fired_at: int | None  # index in the (unsampled) traffic stream where drift was first detected
    mean: float
    lo: float
    hi: float


class DriftMonitor:
    def __init__(
        self,
        baseline: float,
        alpha: float = 0.05,
        min_n: int = 20,
        sample_rate: float = 1.0,
    ) -> None:
        self.baseline = baseline
        self.alpha = alpha
        self.min_n = min_n
        self.sample_rate = sample_rate
        self._obs: list[float] = []
        self._seen = 0
        self._sample_acc = 0.0
        self._fired = False
        self._fired_at: int | None = None
        self._last = DriftAlert(False, None, 0.0, 0.0, 1.0)

    @property
    def n_sampled(self) -> int:
        return len(self._obs)

    def observe(self, outcome: float) -> DriftAlert:
        """Record one live outcome (0..1, e.g. success). Deterministically sampled."""
        self._seen += 1
        self._sample_acc += self.sample_rate
        if self._sample_acc < 1.0:
            return self._last  # not sampled on this trace
        self._sample_acc -= 1.0
        self._obs.append(float(outcome))

        if len(self._obs) >= self.min_n:
            cs = anytime_valid_sequence(self._obs, alpha=self.alpha)
            mean, lo, hi = float(cs.mean[-1]), float(cs.lo[-1]), float(cs.hi[-1])
            if not self._fired and hi < self.baseline:  # upper bound below baseline => real drop
                self._fired = True
                self._fired_at = self._seen
            self._last = DriftAlert(self._fired, self._fired_at, mean, lo, hi)
        return self._last

    def run(self, stream) -> DriftAlert:
        for outcome in stream:
            self.observe(float(outcome))
        return self._last
