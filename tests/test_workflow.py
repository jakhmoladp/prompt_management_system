"""Guards the rule that only a person can change main."""

import yaml

from conftest import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "prompt-pipeline.yml"


def _load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_does_not_run_on_pushes_to_main():
    # PyYAML reads the bare `on:` key as the boolean True.
    assert _load()[True]["push"]["branches"] == ["prompt/**"]


def test_job_refuses_to_run_outside_prompt_branches():
    assert _load()["jobs"]["evaluate"]["if"] == "startsWith(github.ref, 'refs/heads/prompt/')"


def test_workflow_only_pushes_to_candidate_branches_and_never_merges():
    scripts = [step.get("run", "") for step in _load()["jobs"]["evaluate"]["steps"]]
    pushes = [line.strip() for s in scripts for line in s.splitlines() if "git push" in line]
    assert pushes == [
        'git push origin --delete "$candidate"',
        'git push --force origin "HEAD:refs/heads/$candidate"',
    ]
    assert not any("gh pr merge" in s or "gh pr review" in s for s in scripts)
