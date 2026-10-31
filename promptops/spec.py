"""Load and validate what a user pushes: prompts/<name>/prompt.yaml and dataset.jsonl."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")
EXPECT_KEYS = {"tool", "not_tools", "args", "contains", "contains_any", "not_contains", "regex"}
SPEC_KEYS = {"name", "description", "model", "prompt", "tools", "params", "refine", "checks"}


class SpecError(ValueError):
    """The pushed prompt, dataset or checks are invalid."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Refine:
    enabled: bool = True
    reflection_model: str | None = None
    max_metric_calls: int = 60


@dataclass(frozen=True)
class Checks:
    min_pass_rate: float = 0.8
    max_regression: float = 0.0
    max_critical_failures: int = 0
    max_prompt_growth: float = 3.0
    preserve_placeholders: bool = True
    max_avg_latency_ms: float | None = None


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    input: str
    expect: dict[str, Any]
    vars: dict[str, str] = field(default_factory=dict)
    critical: bool = False


@dataclass(frozen=True)
class PromptSpec:
    name: str
    description: str
    model: str
    prompt: str
    tools: tuple[Tool, ...]
    max_tokens: int
    refine: Refine
    checks: Checks
    cases: tuple[Case, ...]

    def split(self, split: str) -> list[Case]:
        return [case for case in self.cases if case.split == split]

    def content_hash(self) -> str:
        """Identifies a submission; line endings and YAML comments don't change it."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def placeholders(prompt: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(prompt))


def render(prompt: str, variables: dict[str, str]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: str(variables.get(m.group(1), m.group(0))), prompt)


def discover(prompts_dir: Path) -> list[Path]:
    if not prompts_dir.is_dir():
        return []
    return sorted(p for p in prompts_dir.iterdir() if (p / "prompt.yaml").is_file())


def load_spec(prompt_dir: Path) -> PromptSpec:
    try:
        raw = yaml.safe_load((prompt_dir / "prompt.yaml").read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{prompt_dir.name}: prompt.yaml is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{prompt_dir.name}: prompt.yaml must be a mapping")

    name = prompt_dir.name
    _reject_unknown(raw, SPEC_KEYS, f"{name}: prompt.yaml")
    if not NAME_RE.match(name):
        raise SpecError(f"{name}: folder name must match {NAME_RE.pattern}")
    if raw.get("name", name) != name:
        raise SpecError(f"{name}: 'name' ({raw['name']}) must match the folder name")
    for key in ("model", "prompt"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise SpecError(f"{name}: '{key}' is required and must be a non-empty string")
    if "/" not in raw["model"]:
        raise SpecError(f"{name}: model must be 'provider/model', got '{raw['model']}'")

    params = raw.get("params") or {}
    _reject_unknown(params, {"max_tokens"}, f"{name}: params")
    tools = _load_tools(name, raw.get("tools") or [])
    spec = PromptSpec(
        name=name,
        description=str(raw.get("description", "")),
        model=raw["model"],
        prompt=raw["prompt"].strip().replace("\r\n", "\n"),
        tools=tools,
        max_tokens=int(params.get("max_tokens", 1024)),
        refine=_load_section(Refine, raw.get("refine"), f"{name}: refine"),
        checks=_load_section(Checks, raw.get("checks"), f"{name}: checks"),
        cases=_load_cases(prompt_dir / "dataset.jsonl", name),
    )
    _validate_cases(spec)
    return spec


def _reject_unknown(section: dict, allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _load_section(cls, raw: dict | None, where: str):
    raw = raw or {}
    _reject_unknown(raw, set(cls.__dataclass_fields__), where)
    return cls(**raw)


def _load_tools(name: str, raw_tools: list[dict]) -> tuple[Tool, ...]:
    tools = []
    for raw in raw_tools:
        if not isinstance(raw, dict) or not raw.get("name") or not raw.get("description"):
            raise SpecError(f"{name}: every tool needs a 'name' and a 'description'")
        _reject_unknown(raw, {"name", "description", "parameters"}, f"{name}: tool {raw['name']}")
        parameters = raw.get("parameters") or {"type": "object", "properties": {}}
        tools.append(Tool(raw["name"], raw["description"].strip(), parameters))
    if len({t.name for t in tools}) != len(tools):
        raise SpecError(f"{name}: tool names must be unique")
    return tuple(tools)


def _load_cases(path: Path, name: str) -> tuple[Case, ...]:
    if not path.is_file():
        raise SpecError(f"{name}: dataset.jsonl is missing")
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{name}: dataset.jsonl line {number}"
        try:
            row = json.loads(line)
            cases.append(Case(
                id=row["id"],
                split=row["split"],
                input=row["input"],
                expect=row["expect"],
                vars=row.get("vars", {}),
                critical=bool(row.get("critical", False)),
            ))
        except json.JSONDecodeError as exc:
            raise SpecError(f"{where}: invalid JSON: {exc}") from exc
        except KeyError as exc:
            raise SpecError(f"{where}: missing field {exc}") from exc
    return tuple(cases)


def _validate_cases(spec: PromptSpec) -> None:
    ids = [case.id for case in spec.cases]
    if len(set(ids)) != len(ids):
        raise SpecError(f"{spec.name}: case ids must be unique")
    if not spec.split("test"):
        raise SpecError(f"{spec.name}: dataset needs at least one case with split 'test'")
    if spec.refine.enabled and not spec.split("train"):
        raise SpecError(f"{spec.name}: refinement needs at least one case with split 'train'")

    tool_names = {tool.name for tool in spec.tools}
    required_vars = placeholders(spec.prompt)
    for case in spec.cases:
        where = f"{spec.name}: case {case.id}"
        if case.split not in {"train", "test"}:
            raise SpecError(f"{where}: split must be 'train' or 'test'")
        # A misspelled or empty expectation would pass silently and weaken the gate.
        _reject_unknown(case.expect, EXPECT_KEYS, f"{where}: expect")
        if not case.expect:
            raise SpecError(f"{where}: expect must contain at least one check")
        if "args" in case.expect and not case.expect.get("tool"):
            raise SpecError(f"{where}: 'args' requires 'tool'")
        named = _as_list(case.expect.get("tool")) + _as_list(case.expect.get("not_tools"))
        unknown_tools = set(named) - tool_names
        if unknown_tools:
            raise SpecError(f"{where}: unknown tools {sorted(unknown_tools)}")
        missing = required_vars - set(case.vars)
        if missing:
            raise SpecError(f"{where}: missing vars for placeholders {sorted(missing)}")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
