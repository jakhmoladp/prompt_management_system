"""A fake model behind DSPy's real LM and GEPA code paths, so tests need no network or keys."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import dspy
import pytest
import yaml
from dspy import lm15

REPO_ROOT = Path(__file__).resolve().parent.parent
GOOD_MARKER = "Follow the refund limit strictly."


def behave(system: str, user: str, offered_tools: set[str]) -> tuple[str, list[tuple[str, dict]]]:
    """Scripted agent: without GOOD_MARKER it issues large refunds, which fails critical cases."""
    text = user.lower()
    order = re.search(r"\b[a-z]\d{4}\b", user, re.IGNORECASE)
    order_id = order.group(0).upper() if order else ""
    amount = re.search(r"\$([\d,]+)", user)
    dollars = int(amount.group(1).replace(",", "")) if amount else 0

    if "refund" in text or "money back" in text or "all of it back" in text:
        if dollars > 100 and GOOD_MARKER in system:
            return "", [("escalate_to_human", {"reason": "over limit"})]
        return "", [("issue_refund", {"order_id": order_id, "amount": dollars})]
    if "human" in text or "real person" in text:
        return "", [("escalate_to_human", {"reason": "requested"})]
    if order_id:
        return "", [("lookup_order", {"order_id": order_id})]
    if "return" in text or "ship" in text:
        return "", [("search_help_center", {"query": user})]
    if "which store" in text:
        match = re.search(r"support agent for (.+?)\.", system)
        return f"You're chatting with {match.group(1) if match else 'us'}.", []
    if "order" in text or "delivery" in text:
        return "What is your order number?", []
    return "I can only help with your orders. How can I help?", []


class FakeEngine:
    """Implements DSPy's custom engine contract, complete(Request) -> Response."""

    def __init__(self, marker: str = GOOD_MARKER, fail: bool = False):
        self.marker = marker
        self.fail = fail
        self.requests: list[lm15.Request] = []

    def complete(self, request: lm15.Request) -> lm15.Response:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("provider is down")
        user_text = "\n".join(part.text for m in request.messages for part in m.parts_of(lm15.TextPart))
        system = request.system if isinstance(request.system, str) else ""
        if request.tools:
            text, calls = behave(system, user_text, {t.name for t in request.tools})
            return self._response(text, calls)
        if "new_instruction" in system:
            return self._response(self._reflection(user_text), [])
        return self._response(self._dspy_turn(system, user_text), [])

    def _reflection(self, user_text: str) -> str:
        """GEPA's reflection step: rewrite the current instruction."""
        current = re.search(r"\[\[ ## current_instruction ## \]\]\n(.*?)\n\n\[\[", user_text, re.DOTALL)
        return json.dumps({"new_instruction": f"{current.group(1)}\n{self.marker}"})

    def _dspy_turn(self, system: str, user_text: str) -> str:
        """A run of the prompt inside the optimizer, answered in DSPy's JSON field format."""
        message = re.search(r"\[\[ ## user_message ## \]\]\n(.*?)\n\n\[\[", user_text, re.DOTALL)
        reply, calls = behave(system, message.group(1), set())
        name, args = calls[0] if calls else ("none", {})
        return json.dumps({"response": reply, "tool_name": name, "tool_args_json": json.dumps(args)})

    def _response(self, text: str, calls: list[tuple[str, dict]]) -> lm15.Response:
        parts = [lm15.TextPart(text=text)] if text else []
        parts += [lm15.ToolCallPart(id=f"call_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
        return lm15.Response(
            id="fake",
            model="fake",
            message=lm15.Message(role="assistant", parts=tuple(parts)),
            finish_reason="tool_call" if calls else "stop",
            usage=lm15.Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def lm_factory(engine):
    return lambda model: dspy.LM(model, engine=engine, cache=False, num_retries=0)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A copy of the shipped sample prompt with a small refinement budget."""
    target = tmp_path / "prompts" / "support-agent"
    shutil.copytree(REPO_ROOT / "prompts" / "support-agent", target)
    return tmp_path


def edit_prompt_yaml(workspace: Path, **changes) -> None:
    path = workspace / "prompts" / "support-agent" / "prompt.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
