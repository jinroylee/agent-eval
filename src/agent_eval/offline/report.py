"""Render a SuiteResult as machine (JSON / JUnit XML) and human (text) reports.

The JUnit output drops straight into CI dashboards; the JSON is the durable artifact; the text is
what the CLI prints. A failed threshold becomes a JUnit ``<failure>`` so CI surfaces it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from xml.sax.saxutils import escape

from agent_eval.core.suite import SuiteResult


def _aggregate_dict(a) -> dict:
    d = {
        "metric": a.metric,
        "value": a.value,
        "ci_low": a.ci_low,
        "ci_high": a.ci_high,
        "n": a.n,
        "n_errors": a.n_errors,
        "aggregation": a.aggregation.value,
        "higher_is_better": a.higher_is_better,
    }
    if a.breakdown:  # only when a criteria-based metric produced one — payloads otherwise unchanged
        d["breakdown"] = dict(a.breakdown)
    return d


def to_dict(result: SuiteResult) -> dict:
    return {
        "agent_type": result.agent_type,
        "category": result.category,
        "n_items": result.n_items,
        "verdict": {
            "passed": result.verdict.passed,
            "metric_passed": result.verdict.metric_passed,
            "reasons": result.verdict.reasons,
        },
        "aggregates": [_aggregate_dict(a) for a in result.aggregates],
    }


def to_json(result: SuiteResult) -> str:
    return json.dumps(to_dict(result), indent=2)


def to_text(result: SuiteResult) -> str:
    lines = [f"=== {result.agent_type}/{result.category}  n={result.n_items} ==="]
    for a in result.aggregates:
        gate = "PASS" if result.verdict.metric_passed.get(a.metric) else "FAIL"
        lines.append(
            f"{a.metric:<24} {a.value:>7.3f}  [{a.ci_low:.3f}, {a.ci_high:.3f}]  {gate}"
        )
        if a.breakdown:
            lines.extend(f"    · {name:<20} {value:>7.3f}" for name, value in a.breakdown.items())
    lines.append(f"VERDICT: {'PASS' if result.verdict.passed else 'FAIL'}")
    lines.extend(f"  - {r}" for r in result.verdict.reasons)
    return "\n".join(lines)


def to_junit(results: Sequence[SuiteResult]) -> str:
    total_failures = sum(0 if r.verdict.passed else 1 for r in results)
    out = [f'<testsuites failures="{total_failures}">']
    for res in results:
        n_fail = sum(
            1 for a in res.aggregates if res.verdict.metric_passed.get(a.metric) is False
        )
        name = escape(f"{res.agent_type}/{res.category}")
        out.append(f'<testsuite name="{name}" tests="{len(res.aggregates)}" failures="{n_fail}">')
        for a in res.aggregates:
            out.append(f'<testcase classname="{escape(res.category)}" name="{escape(a.metric)}">')
            if res.verdict.metric_passed.get(a.metric) is False:
                out.append(
                    f'<failure message="below threshold">value={a.value:.3f} '
                    f"CI=[{a.ci_low:.3f}, {a.ci_high:.3f}] n={a.n}</failure>"
                )
            out.append("</testcase>")
        out.append("</testsuite>")
    out.append("</testsuites>")
    return "\n".join(out)
