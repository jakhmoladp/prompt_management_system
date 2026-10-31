"""End to end through the CLI: real DSPy LM, real GEPA, fake provider engine."""

import json

import dspy
import yaml

from promptops.cli import EXIT_ERROR, EXIT_OK, EXIT_REJECTED, main
from promptops.registry import Registry

from conftest import GOOD_MARKER, FakeEngine, edit_prompt_yaml

REFINE_OFF = {"enabled": False}


def run(workspace, lm_factory, *extra):
    return main([
        "--prompts-dir", str(workspace / "prompts"),
        "--registry-dir", str(workspace / "registry"),
        *extra,
    ], lm_factory=lm_factory)


def registry(workspace) -> Registry:
    return Registry(workspace / "registry")


def good_prompt(workspace) -> str:
    raw = yaml.safe_load((workspace / "prompts/support-agent/prompt.yaml").read_text(encoding="utf-8"))
    return raw["prompt"] + GOOD_MARKER


def test_gepa_refines_and_first_version_is_approved(workspace, lm_factory):
    assert run(workspace, lm_factory) == EXIT_OK

    reg = registry(workspace)
    approved = reg.approved("support-agent")
    assert approved["version"] == 1
    assert approved["status"] == "approved"
    assert approved["refined"] is True
    assert GOOD_MARKER in approved["prompt"]
    assert GOOD_MARKER not in approved["submitted_prompt"]
    assert "{{store_name}}" in approved["prompt"]
    assert reg.entry("support-agent")["approved_version"] == 1
    assert (workspace / "registry/support-agent/versions/v1.yaml").is_file()

    report = json.loads((workspace / "registry/support-agent/reports/v1.json").read_text(encoding="utf-8"))
    assert report["baseline"] is None
    assert report["candidate"]["total"] == 15
    assert {case["id"] for case in report["candidate"]["cases"]} == {
        json.loads(line)["id"]
        for line in (workspace / "prompts/support-agent/dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "test"
    }


def test_rejected_candidate_is_recorded_but_not_approved(workspace, lm_factory):
    edit_prompt_yaml(workspace, refine=REFINE_OFF)  # Unrefined, it issues large refunds.
    assert run(workspace, lm_factory) == EXIT_REJECTED

    reg = registry(workspace)
    assert reg.approved("support-agent") is None
    assert reg.entry("support-agent")["approved_version"] is None
    assert reg.entry("support-agent")["history"][0]["status"] == "rejected"
    assert (workspace / "registry/support-agent/versions/v1.yaml").is_file()


def test_unchanged_submission_is_skipped_without_model_calls(workspace, lm_factory, engine):
    assert run(workspace, lm_factory) == EXIT_OK
    engine.requests.clear()

    assert run(workspace, lm_factory) == EXIT_OK
    assert engine.requests == []
    assert len(registry(workspace).entry("support-agent")["history"]) == 1


def test_force_reevaluates_unchanged_submission(workspace, lm_factory):
    assert run(workspace, lm_factory) == EXIT_OK
    assert run(workspace, lm_factory, "--force") == EXIT_OK
    assert registry(workspace).entry("support-agent")["approved_version"] == 2


def test_regression_against_approved_keeps_previous_version(workspace, lm_factory):
    edit_prompt_yaml(workspace, prompt=good_prompt(workspace), refine=REFINE_OFF)
    assert run(workspace, lm_factory) == EXIT_OK

    # Drop the refund rule, and lower the bar so that only the comparison can catch it.
    edit_prompt_yaml(workspace, prompt=good_prompt(workspace).replace(GOOD_MARKER, ""), checks={
        "min_pass_rate": 0.1, "max_critical_failures": 99,
    })
    edit_prompt_yaml(workspace, refine=REFINE_OFF)
    assert run(workspace, lm_factory) == EXIT_REJECTED

    reg = registry(workspace)
    assert reg.approved("support-agent")["version"] == 1
    assert reg.entry("support-agent")["approved_version"] == 1
    assert [h["status"] for h in reg.entry("support-agent")["history"]] == ["approved", "rejected"]

    report = json.loads((workspace / "registry/support-agent/reports/v2.json").read_text(encoding="utf-8"))
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"no_regression"}
    assert "test-large-refund" in report["comparison"]["newly_failing"]
    assert report["comparison"]["against"] == "approved v1"
    assert report["comparison"]["prompt_diff"]


def test_improvement_moves_approved_pointer(workspace, lm_factory):
    edit_prompt_yaml(workspace, prompt=good_prompt(workspace), refine=REFINE_OFF)
    assert run(workspace, lm_factory) == EXIT_OK
    edit_prompt_yaml(workspace, prompt=good_prompt(workspace) + "\nBe concise.")
    assert run(workspace, lm_factory) == EXIT_OK

    approved = registry(workspace).approved("support-agent")
    assert approved["version"] == 2
    assert approved["parent_version"] == 1
    assert "Be concise." in approved["prompt"]


def test_refiner_dropping_placeholder_is_rejected(workspace):
    class Forgetful(FakeEngine):
        def _response(self, text, calls):
            return super()._response(text.replace("{{store_name}}", "the store"), calls)

    engine = Forgetful()
    factory = lambda model: dspy.LM(model, engine=engine, cache=False, num_retries=0)
    assert run(workspace, factory) == EXIT_REJECTED
    report = json.loads((workspace / "registry/support-agent/reports/v1.json").read_text(encoding="utf-8"))
    assert {c["name"] for c in report["checks"] if not c["passed"]} == {"preserve_placeholders"}


def test_provider_outage_records_nothing_and_is_retried(workspace, lm_factory):
    edit_prompt_yaml(workspace, refine=REFINE_OFF, prompt=good_prompt(workspace))
    down = FakeEngine(fail=True)
    assert run(workspace, lambda m: dspy.LM(m, engine=down, cache=False, num_retries=0)) == EXIT_ERROR
    assert not (workspace / "registry").exists()

    assert run(workspace, lm_factory) == EXIT_OK
    assert registry(workspace).approved("support-agent")["version"] == 1


def test_baseline_outage_does_not_approve(workspace, lm_factory):
    edit_prompt_yaml(workspace, refine=REFINE_OFF, prompt=good_prompt(workspace))
    assert run(workspace, lm_factory) == EXIT_OK
    edit_prompt_yaml(workspace, prompt=good_prompt(workspace) + "\nBe concise.")

    down = FakeEngine(fail=True)
    assert run(workspace, lambda m: dspy.LM(m, engine=down, cache=False, num_retries=0)) == EXIT_ERROR
    reg = registry(workspace)
    assert reg.approved("support-agent")["version"] == 1
    assert len(reg.entry("support-agent")["history"]) == 1


def test_invalid_submission_does_not_block_other_prompts(workspace, lm_factory):
    broken = workspace / "prompts" / "broken"
    broken.mkdir()
    (broken / "prompt.yaml").write_text("model: openai/x\n", encoding="utf-8")
    assert run(workspace, lm_factory) == EXIT_ERROR
    assert registry(workspace).approved("support-agent")["version"] == 1
    assert registry(workspace).entry("broken") is None


def test_prompt_filter(workspace, lm_factory):
    assert run(workspace, lm_factory, "--prompt", "does-not-exist") == EXIT_ERROR
    assert run(workspace, lm_factory, "--prompt", "support-agent") == EXIT_OK


def test_tools_reach_the_model_natively_with_rendered_prompt(workspace, lm_factory, engine):
    edit_prompt_yaml(workspace, refine=REFINE_OFF)
    run(workspace, lm_factory)
    request = engine.requests[0]
    assert {tool.name for tool in request.tools} == {
        "lookup_order", "issue_refund", "escalate_to_human", "search_help_center",
    }
    assert "{{" not in request.system
    assert request.config.temperature is None
