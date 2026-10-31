import json
from pathlib import Path

import pytest

from promptops.evaluate import CaseResult, EvalResult, score_case
from promptops.gate import run_gate
from promptops.spec import Checks, SpecError, load_spec, placeholders, render

from conftest import REPO_ROOT


def _call(name, **args):
    return {"name": name, "args": args}


@pytest.mark.parametrize("expect, text, calls, failed", [
    ({"tool": "lookup"}, "", [_call("lookup")], False),
    ({"tool": "lookup"}, "", [], True),
    ({"tool": ["a", "b"]}, "", [_call("b")], False),
    ({"tool": None}, "hi", [], False),
    ({"tool": None}, "", [_call("a")], True),
    ({"not_tools": ["refund"]}, "", [_call("refund")], True),
    ({"tool": "a", "args": {"id": "x1"}}, "", [_call("a", id="X1")], False),
    ({"tool": "a", "args": {"id": "x1"}}, "", [_call("a", id="other")], True),
    ({"tool": "a", "args": {"amount": 40}}, "", [_call("a", amount=40)], False),
    ({"contains": ["Acme"]}, "welcome to acme", [], False),
    ({"contains": ["Acme"]}, "welcome", [], True),
    ({"contains_any": ["x", "y"]}, "has y", [], False),
    ({"contains_any": ["x", "y"]}, "none", [], True),
    ({"not_contains": ["SECRET"]}, "the secret", [], True),
    ({"not_contains": ["SECRET"]}, "", [], False),
    ({"regex": r"order (id|number)"}, "your Order Number?", [], False),
    ({"regex": r"order (id|number)"}, "hello", [], True),
])
def test_score_case(expect, text, calls, failed):
    assert bool(score_case(expect, text, calls)) is failed


def test_empty_response_does_not_crash_text_checks():
    assert score_case({"contains": ["x"]}, None, [])
    assert not score_case({"not_contains": ["x"]}, None, [])


def test_render_and_placeholders():
    assert placeholders("a {{x}} b {{ y }}") == {"x", "y"}
    assert render("hi {{ x }} {{unset}}", {"x": "there"}) == "hi there {{unset}}"


def _eval(passed: int, failed: int, critical_failed: int = 0) -> EvalResult:
    def case(i, ok, critical=False):
        return CaseResult(f"c{i}", ok, critical, (), "", (), None, 100.0, 1, 1)
    cases = [case(i, True) for i in range(passed)]
    cases += [case(100 + i, False) for i in range(failed)]
    cases += [case(200 + i, False, critical=True) for i in range(critical_failed)]
    return EvalResult("m", tuple(cases))


def _gate(candidate, baseline=None, checks=Checks(), candidate_prompt="p", submitted="p"):
    return {c.name: c.passed for c in run_gate(checks, candidate, baseline, candidate_prompt, submitted)}


def test_gate_passes_good_candidate():
    assert all(_gate(_eval(9, 1), _eval(8, 2)).values())


def test_gate_fails_below_minimum_pass_rate():
    assert _gate(_eval(7, 3))["min_pass_rate"] is False


def test_gate_fails_on_regression_even_above_minimum():
    assert _gate(_eval(8, 2), _eval(9, 1))["no_regression"] is False


def test_gate_regression_tolerance_is_respected():
    assert _gate(_eval(8, 2), _eval(9, 1), Checks(max_regression=0.1))["no_regression"] is True


def test_gate_fails_on_any_critical_failure():
    assert _gate(_eval(19, 0, critical_failed=1))["max_critical_failures"] is False


def test_gate_fails_when_placeholder_lost():
    assert _gate(_eval(10, 0), candidate_prompt="hi", submitted="hi {{x}}")["preserve_placeholders"] is False


def test_gate_fails_on_prompt_bloat():
    assert _gate(_eval(10, 0), candidate_prompt="x" * 40, submitted="x" * 10)["max_prompt_growth"] is False


def test_gate_latency_check_only_runs_when_configured():
    assert "max_avg_latency_ms" not in _gate(_eval(10, 0))
    assert _gate(_eval(10, 0), checks=Checks(max_avg_latency_ms=50))["max_avg_latency_ms"] is False


def _write_spec(tmp_path: Path, yaml_text: str, rows: list[dict]) -> Path:
    target = tmp_path / "demo"
    target.mkdir()
    (target / "prompt.yaml").write_text(yaml_text, encoding="utf-8")
    (target / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return target


BASE_YAML = "model: openai/x\nprompt: hello\nrefine:\n  enabled: false\n"
TEST_ROW = {"id": "a", "split": "test", "input": "hi", "expect": {"contains": ["x"]}}


def test_shipped_sample_is_valid():
    spec = load_spec(REPO_ROOT / "prompts" / "support-agent")
    assert len(spec.split("train")) == 10
    assert len(spec.split("test")) == 15
    assert any(case.critical for case in spec.split("test"))


@pytest.mark.parametrize("yaml_text, rows, message", [
    ("prompt: hi\n", [TEST_ROW], "'model' is required"),
    ("model: nomodelprefix\nprompt: hi\n", [TEST_ROW], "provider/model"),
    (BASE_YAML + "typo: 1\n", [TEST_ROW], "unknown keys"),
    (BASE_YAML + "checks:\n  min_passrate: 0.5\n", [TEST_ROW], "unknown keys"),
    (BASE_YAML, [{**TEST_ROW, "split": "train"}], "split 'test'"),
    (BASE_YAML, [TEST_ROW, TEST_ROW], "unique"),
    (BASE_YAML, [{**TEST_ROW, "expect": {"contain": ["x"]}}], "unknown keys"),
    (BASE_YAML, [{**TEST_ROW, "expect": {}}], "at least one check"),
    (BASE_YAML, [{**TEST_ROW, "expect": {"tool": "ghost"}}], "unknown tools"),
    (BASE_YAML.replace("hello", "hi {{who}}"), [TEST_ROW], "missing vars"),
    ("model: openai/x\nprompt: hi\n", [TEST_ROW], "split 'train'"),
])
def test_invalid_submissions_are_rejected(tmp_path, yaml_text, rows, message):
    with pytest.raises(SpecError, match=message):
        load_spec(_write_spec(tmp_path, yaml_text, rows))


def test_content_hash_ignores_comments_and_line_endings(tmp_path):
    first = load_spec(_write_spec(tmp_path, BASE_YAML, [TEST_ROW]))
    path = tmp_path / "demo" / "prompt.yaml"
    path.write_bytes(("# a comment\n" + BASE_YAML).replace("\n", "\r\n").encode("utf-8"))
    assert load_spec(tmp_path / "demo").content_hash() == first.content_hash()
    path.write_text(BASE_YAML.replace("hello", "changed"), encoding="utf-8")
    assert load_spec(tmp_path / "demo").content_hash() != first.content_hash()
