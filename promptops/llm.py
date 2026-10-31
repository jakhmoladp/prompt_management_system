"""The single place that talks to a model: a real system prompt with native tools."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import dspy
from dspy import lm15

from promptops.spec import Tool

LMFactory = Callable[[str], dspy.LM]


@dataclass(frozen=True)
class ModelOutput:
    text: str
    tool_calls: tuple[dict, ...]
    finish_reason: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int


def make_lm(model: str) -> dspy.LM:
    # Sampling parameters are left unset: current models reject them. Results are not cached,
    # so a re-evaluated baseline measures the model as it behaves today.
    return dspy.LM(model, cache=False, num_retries=3)


def call_model(lm: dspy.LM, system: str, user: str, tools: Sequence[Tool], max_tokens: int) -> ModelOutput:
    request = lm15.Request(
        model=lm.model,
        system=system,
        messages=(lm15.Message.user(user),),
        tools=tuple(lm15.FunctionTool(name=t.name, description=t.description, parameters=t.parameters) for t in tools),
        config=lm15.Config(max_tokens=max_tokens),
    )
    started = time.perf_counter()
    response = lm(request)
    latency_ms = (time.perf_counter() - started) * 1000
    # Response.text is empty when a turn mixes text with tool calls, so read the parts.
    text = "".join(part.text for part in response.message.parts_of(lm15.TextPart))
    usage = response.usage
    return ModelOutput(
        text=text,
        tool_calls=tuple({"name": call.name, "args": call.input} for call in response.tool_calls),
        finish_reason=response.finish_reason,
        latency_ms=latency_ms,
        input_tokens=(usage.input_tokens or 0) if usage else 0,
        output_tokens=(usage.output_tokens or 0) if usage else 0,
    )
