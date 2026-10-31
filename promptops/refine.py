"""Generate the candidate prompt with DSPy's GEPA optimizer.

GEPA sees only the train split. It starts from the submitted prompt and reflects on the failure
messages from `score_case`, so it can return the submitted prompt unchanged if nothing beats it.
"""

from __future__ import annotations

import json

import dspy

from promptops.evaluate import score_case
from promptops.llm import LMFactory
from promptops.spec import Case, PromptSpec, Tool


class AgentTurn(dspy.Signature):
    user_message: str = dspy.InputField()
    available_tools: str = dspy.InputField(desc="JSON list of tools that can be called")
    template_variables: str = dspy.InputField(desc="JSON values for {{placeholders}} in the instructions")
    response: str = dspy.OutputField(desc="Text reply to the user; may be empty when calling a tool")
    tool_name: str = dspy.OutputField(desc="Name of the tool to call, or 'none'")
    tool_args_json: str = dspy.OutputField(desc="JSON object of arguments for the tool, or {}")


class PromptProgram(dspy.Module):
    def __init__(self, prompt: str):
        super().__init__()
        self.predict = dspy.Predict(AgentTurn.with_instructions(prompt))

    def forward(self, user_message: str, available_tools: str, template_variables: str) -> dspy.Prediction:
        return self.predict(
            user_message=user_message,
            available_tools=available_tools,
            template_variables=template_variables,
        )


def _tool_calls(prediction: dspy.Prediction) -> list[dict]:
    name = (prediction.tool_name or "").strip()
    if not name or name.lower() == "none":
        return []
    try:
        args = json.loads(prediction.tool_args_json or "{}")
    except json.JSONDecodeError:
        args = {}
    return [{"name": name, "args": args if isinstance(args, dict) else {}}]


def metric(gold, prediction, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    failures = score_case(gold.expect, prediction.response, _tool_calls(prediction))
    return dspy.Prediction(
        score=0.0 if failures else 1.0,
        feedback="; ".join(failures) or "All expectations met.",
    )


def _to_example(case: Case, tools: tuple[Tool, ...]) -> dspy.Example:
    return dspy.Example(
        user_message=case.input,
        available_tools=json.dumps([vars(tool) for tool in tools]),
        template_variables=json.dumps(case.vars),
        expect=case.expect,
    ).with_inputs("user_message", "available_tools", "template_variables")


def refine_prompt(spec: PromptSpec, lm_factory: LMFactory) -> str:
    """Return the best prompt GEPA finds on the train split."""
    if not spec.refine.enabled:
        return spec.prompt
    trainset = [_to_example(case, spec.tools) for case in spec.split("train")]
    optimizer = dspy.GEPA(
        metric=metric,
        max_metric_calls=spec.refine.max_metric_calls,
        reflection_lm=lm_factory(spec.refine.reflection_model or spec.model),
        num_threads=4,
        track_stats=False,
    )
    with dspy.context(lm=lm_factory(spec.model)):
        program = optimizer.compile(PromptProgram(spec.prompt), trainset=trainset)
    return program.predict.signature.instructions.strip()
