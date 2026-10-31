"""One prompt's journey: refine, evaluate against the approved version, gate, record."""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from typing import Any

from promptops.evaluate import EvalResult, evaluate
from promptops.gate import run_gate
from promptops.llm import LMFactory
from promptops.refine import refine_prompt
from promptops.registry import Registry
from promptops.spec import PromptSpec, Tool


@dataclass(frozen=True)
class RunContext:
    """Where this run came from, recorded on every version."""

    created_at: str
    commit: str | None = None
    actor: str | None = None


@dataclass(frozen=True)
class Outcome:
    name: str
    status: str  # approved | rejected | skipped
    version: int | None = None
    report: dict[str, Any] | None = None


def process_prompt(
    spec: PromptSpec,
    registry: Registry,
    lm_factory: LMFactory,
    context: RunContext,
    force: bool = False,
) -> Outcome:
    content_hash = spec.content_hash()
    if not force and registry.latest_hash(spec.name) == content_hash:
        return Outcome(spec.name, "skipped")

    test_cases = spec.split("test")
    candidate_prompt = refine_prompt(spec, lm_factory)
    candidate = evaluate(lm_factory(spec.model), candidate_prompt, spec.tools, test_cases, spec.max_tokens)

    # The baseline runs again rather than reusing its stored score: the dataset or the model's
    # behaviour may have changed since it was approved, and only the same run is comparable.
    approved = registry.approved(spec.name)
    baseline = None
    if approved:
        baseline = evaluate(
            lm_factory(approved["model"]),
            approved["prompt"],
            [Tool(**tool) for tool in approved["tools"]],
            test_cases,
            approved["max_tokens"],
        )

    checks = run_gate(spec.checks, candidate, baseline, candidate_prompt, spec.prompt)
    status = "approved" if all(check.passed for check in checks) else "rejected"
    # Rejected candidates get no version: the registry holds only what can reach main.
    version = registry.next_version(spec.name) if status == "approved" else None

    report = {
        "name": spec.name,
        "version": version,
        "status": status,
        "checks": [asdict(check) for check in checks],
        "candidate": candidate.to_dict(),
        "baseline": baseline.to_dict() if baseline else None,
        "comparison": _compare(spec, candidate_prompt, candidate, approved, baseline),
    }
    if status == "rejected":
        return Outcome(spec.name, status, None, report)

    record = {
        "name": spec.name,
        "version": version,
        "status": status,
        "description": spec.description,
        "model": spec.model,
        "max_tokens": spec.max_tokens,
        "prompt": candidate_prompt,
        "tools": [asdict(tool) for tool in spec.tools],
        "submitted_prompt": spec.prompt,
        "refined": candidate_prompt != spec.prompt,
        "parent_version": approved["version"] if approved else None,
        "scores": candidate.summary(),
        "content_hash": content_hash,
        "created_at": context.created_at,
        "commit": context.commit,
        "submitted_by": context.actor,
    }
    registry.record(record, report)
    return Outcome(spec.name, status, version, report)


def _compare(
    spec: PromptSpec,
    candidate_prompt: str,
    candidate: EvalResult,
    approved: dict[str, Any] | None,
    baseline: EvalResult | None,
) -> dict[str, Any]:
    """Prompt text diff against the approved version (or the submission on a first run)."""
    previous = approved["prompt"] if approved else spec.prompt
    label = f"approved v{approved['version']}" if approved else "submitted"
    comparison = {
        "against": label,
        "prompt_diff": "\n".join(difflib.unified_diff(
            previous.splitlines(), candidate_prompt.splitlines(), label, "candidate", lineterm="",
        )),
        "newly_failing": [],
        "newly_passing": [],
    }
    if approved:
        comparison["model_change"] = None if approved["model"] == spec.model else [approved["model"], spec.model]
        comparison["tools_changed"] = approved["tools"] != [asdict(tool) for tool in spec.tools]
    if baseline:
        before = {case.id: case.passed for case in baseline.cases}
        comparison["newly_failing"] = [c.id for c in candidate.cases if before[c.id] and not c.passed]
        comparison["newly_passing"] = [c.id for c in candidate.cases if not before[c.id] and c.passed]
    return comparison


def render_markdown(outcome: Outcome) -> str:
    """Human summary for the GitHub Actions job page."""
    report = outcome.report
    if report is None:
        return f"## {outcome.name}: skipped (already evaluated)\n"
    mark = "✅" if outcome.status == "approved" else "❌"
    label = f"{outcome.name} v{outcome.version}" if outcome.version else outcome.name
    lines = [f"## {mark} {label}: {outcome.status}", ""]
    lines += ["| Check | Result | Detail |", "| --- | --- | --- |"]
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {'pass' if check['passed'] else 'FAIL'} | {check['detail']} |")

    comparison = report["comparison"]
    lines += ["", f"**Compared with:** {comparison['against']}"]
    if comparison["newly_failing"]:
        lines.append(f"**Newly failing:** {', '.join(comparison['newly_failing'])}")
    if comparison["newly_passing"]:
        lines.append(f"**Newly passing:** {', '.join(comparison['newly_passing'])}")

    failed = [case for case in report["candidate"]["cases"] if not case["passed"]]
    if failed:
        lines += ["", "**Failed cases**", ""]
        lines += [f"- `{case['id']}`: {'; '.join(case['failures'])}" for case in failed]
    if comparison["prompt_diff"]:
        lines += ["", "<details><summary>Prompt diff</summary>", "", "```diff",
                  comparison["prompt_diff"], "```", "", "</details>"]
    return "\n".join(lines) + "\n"
