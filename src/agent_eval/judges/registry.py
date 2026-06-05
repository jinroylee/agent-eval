"""Judge registry: pinned configs + their certification records.

``is_certified`` is the gatekeeper — a judge may be used in a gate only if a certification record
exists, passed its floors, AND still matches the judge's current fingerprint (else it's stale).
"""

from __future__ import annotations

from agent_eval.judges.certify import CertificationRecord
from agent_eval.judges.config import JudgeConfig


class JudgeRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, JudgeConfig] = {}
        self._certs: dict[str, CertificationRecord] = {}

    def register(self, config: JudgeConfig) -> None:
        self._configs[config.id] = config

    def get(self, judge_id: str) -> JudgeConfig | None:
        return self._configs.get(judge_id)

    def add_certification(self, record: CertificationRecord) -> None:
        self._certs[record.id] = record

    def get_certification(self, cert_id: str) -> CertificationRecord | None:
        return self._certs.get(cert_id)

    def is_certified(self, cert_id: str | None, fingerprint: str) -> bool:
        if not cert_id:
            return False
        rec = self._certs.get(cert_id)
        return bool(rec and rec.passed and rec.fingerprint == fingerprint)
