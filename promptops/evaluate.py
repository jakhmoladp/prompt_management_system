"""Run a prompt over dataset cases and score each against its expectations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import dspy

from promptops.llm import ModelOutput, call_model
from promptops.spec import Case, Tool, _as_list, render

MAX_WORKERS = 4


class EvaluationError(RuntimeError):
    """A model call failed, so the scores can't be trusted either way."""


@dataclass(frozen=True)
class CaseResult:
    id: str
    passed: bool
    critical: bool
    failures: tuple[str, ...]
    text: str
    tool_calls: tuple[dict, ...]
    finish_reason: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class EvalResult:
    model: str
    cases: tuple[CaseResult, ...]

    @property
    def pass_rate(self) -> float:
        return sum(case.passed for case in self.cases) / len(self.cases)

    @property
    def critical_failures(self) -> list[str]:
        return [case.id for case in self.cases if case.critical and not case.passed]

    @property
    def avg_latency_ms(self) -> float:
        return sum(case.latency_ms for case in self.cases) / len(self.cases)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "pass_rate": round(self.pass_rate, 4),
            "passed": sum(case.passed for case in self.cases),
            "total": len(self.cases),
            "critical_failures": self.critical_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "input_tokens": sum(case.input_tokens for case in self.cases),
            "output_tokens": sum(case.output_tokens for case in self.cases),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "cases": [asdict(case) for case in self.cases]}


def score_case(expect: dict[str, Any], text: str, tool_calls: Sequence[dict]) -> list[str]:
    """Return one message per unmet expectation; an empty list means the case passed."""
    failures = []
    called = [call["name"] for call in tool_calls]
    lowered = (text or "").lower()

    if "tool" in expect:
        accepted = _as_list(expect["tool"])
        if not accepted and called:
            failures.append(f"expected no tool call, got {called}")
        elif accepted and not set(accepted) & set(called):
            failures.append(f"expected a call to one of {accepted}, got {called or 'no tool call'}")
    for forbidden in _as_list(expect.get("not_tools")):
        if forbidden in called:
            failures.append(f"must not call {forbidden}")
    if "args" in expect:
        failures.extend(_score_args(expect, tool_calls))

    for needle in _as_list(expect.get("contains")):
        if needle.lower() not in lowered:
            failures.append(f"response must contain '{needle}'")
    any_of = _as_list(expect.get("contains_any"))
    if any_of and not any(needle.lower() in lowered for needle in any_of):
        failures.append(f"response must contain one of {any_of}")
    for needle in _as_list(expect.get("not_contains")):
        if needle.lower() in lowered:
            failures.append(f"response must not contain '{needle}'")
    if "regex" in expect and not re.search(expect["regex"], text or "", re.IGNORECASE):
        failures.append(f"response must match /{expect['regex']}/")
    return failures


def _score_args(expect: dict[str, Any], tool_calls: Sequence[dict]) -> list[str]:
    accepted = _as_list(expect["tool"])
    matching = [call for call in tool_calls if call["name"] in accepted]
    if not matching:
        return []  # Already reported as a missing tool call.
    wanted = {key: str(value).lower() for key, value in expect["args"].items()}
    for call in matching:
        args = call.get("args") or {}
        if all(str(args.get(key)).lower() == value for key, value in wanted.items()):
            return []
    return [f"expected arguments {expect['args']}, got {[call.get('args') for call in matching]}"]


def evaluate(
    lm: dspy.LM,
    prompt: str,
    tools: Sequence[Tool],
    cases: Sequence[Case],
    max_tokens: int,
) -> EvalResult:
    def run(case: Case) -> CaseResult:
        try:
            output = call_model(lm, render(prompt, case.vars), case.input, tools, max_tokens)
        except Exception as exc:
            raise EvaluationError(f"case {case.id} on {lm.model}: {exc}") from exc
        return _to_result(case, output)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return EvalResult(model=lm.model, cases=tuple(pool.map(run, cases)))


def _to_result(case: Case, output: ModelOutput) -> CaseResult:
    failures = score_case(case.expect, output.text, output.tool_calls)
    return CaseResult(
        id=case.id,
        passed=not failures,
        critical=case.critical,
        failures=tuple(failures),
        text=output.text,
        tool_calls=output.tool_calls,
        finish_reason=output.finish_reason,
        latency_ms=round(output.latency_ms, 1),
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
    )
