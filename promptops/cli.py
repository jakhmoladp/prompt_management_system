"""Command line entry point: python -m promptops [--prompt NAME] [--force]."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from promptops.llm import LMFactory, make_lm
from promptops.pipeline import Outcome, RunContext, process_prompt, render_markdown
from promptops.registry import Registry
from promptops.spec import SpecError, discover, load_spec

EXIT_OK, EXIT_REJECTED, EXIT_ERROR = 0, 1, 2

log = logging.getLogger("promptops")


def main(argv: list[str] | None = None, lm_factory: LMFactory = make_lm) -> int:
    parser = argparse.ArgumentParser(prog="promptops", description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--registry-dir", type=Path, default=Path("registry"))
    parser.add_argument("--prompt", help="Process only this prompt (default: all prompts)")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if unchanged since the last version")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)  # Logs two lines per model call.

    # Every prompt is compared with the registry by content hash instead of diffing commits.
    # Queued runs can be dropped by GitHub; the next run still picks up whatever they missed.
    dirs = discover(args.prompts_dir)
    if args.prompt:
        dirs = [path for path in dirs if path.name == args.prompt]
        if not dirs:
            log.error("no prompt named %s in %s", args.prompt, args.prompts_dir)
            return EXIT_ERROR

    registry = Registry(args.registry_dir)
    context = RunContext(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit=os.environ.get("GITHUB_SHA"),
        actor=os.environ.get("GITHUB_ACTOR"),
    )

    exit_code = EXIT_OK
    for prompt_dir in dirs:
        try:
            outcome = process_prompt(load_spec(prompt_dir), registry, lm_factory, context, force=args.force)
        except SpecError as exc:
            _publish(f"## ⚠️ {prompt_dir.name}: invalid submission\n\n{exc}\n")
            exit_code = EXIT_ERROR
            continue
        except Exception as exc:
            # Nothing is recorded, so the next run retries this prompt from scratch.
            log.exception("%s failed", prompt_dir.name)
            _publish(f"## ⚠️ {prompt_dir.name}: pipeline error, nothing recorded\n\n{exc}\n")
            exit_code = EXIT_ERROR
            continue
        _publish(render_markdown(outcome))
        exit_code = max(exit_code, _exit_code_for(outcome))
    return exit_code


def _exit_code_for(outcome: Outcome) -> int:
    return EXIT_REJECTED if outcome.status == "rejected" else EXIT_OK


def _publish(markdown: str) -> None:
    sys.stdout.write(markdown + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
