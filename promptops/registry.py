"""The prompt registry: versioned files under registry/, reaching main only through a reviewed PR.

registry/
  index.json                   every prompt's approved version and full history
  <name>/approved.yaml         what applications should load
  <name>/versions/v<N>.yaml    every version that passed the gate
  <name>/reports/v<N>.json     the full evaluation report for that version
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

INDEX_FILE = "index.json"


class _BlockDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockDumper.add_representer(str, _represent_str)


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True)


class Registry:
    def __init__(self, root: Path):
        self.root = root

    def index(self) -> dict[str, Any]:
        path = self.root / INDEX_FILE
        if not path.is_file():
            return {"prompts": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def entry(self, name: str) -> dict[str, Any] | None:
        return self.index()["prompts"].get(name)

    def latest_hash(self, name: str) -> str | None:
        entry = self.entry(name)
        return entry["history"][-1]["content_hash"] if entry else None

    def approved(self, name: str) -> dict[str, Any] | None:
        path = self.root / name / "approved.yaml"
        if not path.is_file():
            return None
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def next_version(self, name: str) -> int:
        entry = self.entry(name)
        return entry["history"][-1]["version"] + 1 if entry else 1

    def record(self, record: dict[str, Any], report: dict[str, Any]) -> None:
        """Write an approved version and move the approved pointer to it."""
        name, version = record["name"], record["version"]
        prompt_root = self.root / name
        (prompt_root / "versions").mkdir(parents=True, exist_ok=True)
        (prompt_root / "reports").mkdir(parents=True, exist_ok=True)

        text = _dump_yaml(record)
        (prompt_root / "versions" / f"v{version}.yaml").write_text(text, encoding="utf-8", newline="\n")
        (prompt_root / "reports" / f"v{version}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        (prompt_root / "approved.yaml").write_text(text, encoding="utf-8", newline="\n")

        index = self.index()
        entry = index["prompts"].setdefault(name, {"approved_version": None, "history": []})
        entry["approved_version"] = version
        entry["history"].append({
            "version": version,
            "status": record["status"],
            "pass_rate": record["scores"]["pass_rate"],
            "model": record["model"],
            "content_hash": record["content_hash"],
            "created_at": record["created_at"],
            "commit": record["commit"],
        })
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / INDEX_FILE).write_text(
            json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
