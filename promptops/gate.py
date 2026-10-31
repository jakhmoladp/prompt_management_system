"""The evaluation gate: every check must pass for a candidate to become approved."""

from __future__ import annotations

from dataclasses import dataclass

from promptops.evaluate import EvalResult
from promptops.spec import Checks, placeholders


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_gate(
    checks: Checks,
    candidate: EvalResult,
    baseline: EvalResult | None,
    candidate_prompt: str,
    submitted_prompt: str,
) -> list[CheckResult]:
    results = [
        CheckResult(
            "min_pass_rate",
            candidate.pass_rate >= checks.min_pass_rate,
            f"{candidate.pass_rate:.1%} (required >= {checks.min_pass_rate:.1%})",
        ),
        CheckResult(
            "max_critical_failures",
            len(candidate.critical_failures) <= checks.max_critical_failures,
            f"{len(candidate.critical_failures)} failed (allowed {checks.max_critical_failures})"
            + (f": {', '.join(candidate.critical_failures)}" if candidate.critical_failures else ""),
        ),
    ]
    if baseline is None:
        results.append(CheckResult("no_regression", True, "no approved version to compare against"))
    else:
        floor = baseline.pass_rate - checks.max_regression
        results.append(CheckResult(
            "no_regression",
            candidate.pass_rate >= floor,
            f"candidate {candidate.pass_rate:.1%} vs approved {baseline.pass_rate:.1%} "
            f"(allowed drop {checks.max_regression:.1%})",
        ))

    limit = len(submitted_prompt) * checks.max_prompt_growth
    results.append(CheckResult(
        "max_prompt_growth",
        len(candidate_prompt) <= limit,
        f"{len(candidate_prompt)} chars (limit {limit:.0f}, {checks.max_prompt_growth}x submitted)",
    ))
    if checks.preserve_placeholders:
        lost = placeholders(submitted_prompt) - placeholders(candidate_prompt)
        results.append(CheckResult(
            "preserve_placeholders",
            not lost,
            f"lost {sorted(lost)}" if lost else "all placeholders kept",
        ))
    if checks.max_avg_latency_ms is not None:
        results.append(CheckResult(
            "max_avg_latency_ms",
            candidate.avg_latency_ms <= checks.max_avg_latency_ms,
            f"{candidate.avg_latency_ms:.0f} ms (limit {checks.max_avg_latency_ms:.0f} ms)",
        ))
    return results
