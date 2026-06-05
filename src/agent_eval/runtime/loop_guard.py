"""Guards against degenerate retry loops (oscillation / no-progress / unbounded retries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopGuard:
    max_retries: int
    _seen: list[Any] = field(default_factory=list)

    def remember(self, output: Any) -> None:
        self._seen.append(output)

    def is_repeat(self, output: Any) -> bool:
        """A regenerated output we've already tried means the critic isn't making progress."""
        return output in self._seen

    def is_exhausted(self, attempt: int) -> bool:
        return attempt >= self.max_retries
