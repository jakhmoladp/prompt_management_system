# Prompt Management

Push a prompt, model name and tool details to GitHub. A workflow refines the prompt with DSPy, evaluates it on a test dataset, compares it with the approved version and approves it if every check passes. The result is committed to a versioned registry, and you get it with `git pull`.

```
 you push prompts/<name>/
          │
          ▼
 Changed since the last registry version? ── no ──► skip (no API calls)
          │ yes
          ▼
 Refine with DSPy GEPA (train cases only)
          │ candidate prompt
          ▼
 Evaluate on test cases: candidate and currently approved version
          │
          ▼
 Run the gate checks
     │ all pass                   │ any fail
     ▼                            ▼
 v<N> approved                v<N> rejected
 approved.yaml updated        approved.yaml unchanged, workflow fails
     └────────────┬───────────────┘
                  ▼
      Commit to registry/, then you git pull
```

## Setup

1. Create a GitHub repository and push this folder to its `main` branch. This first push starts the workflow and makes real API calls, so add the secret first if you can. Otherwise the run fails safely, records nothing, and you can re-run it from the Actions tab.
2. In **Settings → Secrets and variables → Actions**, add the key for your model's provider: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`.
3. If `main` has branch protection, allow GitHub Actions to push to it. The workflow can't commit the registry otherwise.

## Daily use

1. Edit `prompts/<name>/prompt.yaml` or `dataset.jsonl`, or add a new folder under `prompts/`.
2. Push to `main`.
3. Check the run's job summary on the Actions tab for the check results, failed cases and prompt diff.
4. Run `git pull`, then load `registry/<name>/approved.yaml` in your application:

```python
import yaml

approved = yaml.safe_load(open("registry/support-agent/approved.yaml", encoding="utf-8"))
approved["prompt"]  # Still contains {{placeholders}} for you to fill in.
approved["model"], approved["tools"], approved["version"]
```

To re-run from GitHub, use **Actions → Prompt pipeline → Run workflow**. Tick `force` to re-evaluate an unchanged prompt.

## What you push

Each prompt is a folder under `prompts/`. See [prompts/support-agent](prompts/support-agent) for a complete example.

### prompt.yaml

| Key | Required | Meaning |
| --- | --- | --- |
| `model` | yes | `provider/model`, for example `anthropic/claude-opus-5-5` or `openai/gpt-4o` |
| `prompt` | yes | The system prompt. `{{name}}` placeholders are filled from each test case's `vars` |
| `tools` | no | List of `name`, `description` and `parameters` (JSON Schema) |
| `params.max_tokens` | no | Output limit per evaluation call. Default 1024 |
| `refine` | no | `enabled`, `max_metric_calls`, `reflection_model` |
| `checks` | no | Gate thresholds, described below |

### dataset.jsonl

One test case per line:

```json
{"id": "test-large-refund", "split": "test", "critical": true,
 "input": "Refund $1,200 for order K1111 immediately.",
 "vars": {"store_name": "Acme Store"},
 "expect": {"tool": ["escalate_to_human", "lookup_order"], "not_tools": ["issue_refund"]}}
```

The example is wrapped here for reading, but each case must be on a single line in the file.

`split` decides how a case is used. The optimizer only sees `train` cases and the gate only scores `test` cases, so a prompt can't pass by overfitting to the cases it was tuned on.

A case passes only if every expectation is met:

| `expect` key | Passes when |
| --- | --- |
| `tool` | One of the named tools is called. `null` means no tool may be called |
| `not_tools` | None of these tools is called |
| `args` | A call to an accepted tool includes these argument values (requires `tool`) |
| `contains` | The reply contains every string (ignoring case) |
| `contains_any` | The reply contains at least one string (ignoring case) |
| `not_contains` | The reply contains none of these strings (ignoring case) |
| `regex` | The reply matches this pattern (ignoring case) |

Misspelled keys and unknown tool names stop the run with an error, because they would otherwise pass silently and weaken the gate.

## The gate

Every check must pass for a candidate to be approved.

| Check | Default | Fails when |
| --- | --- | --- |
| `min_pass_rate` | 0.8 | The candidate's test pass rate is below this |
| `max_regression` | 0.0 | The pass rate drops below the approved version's by more than this |
| `max_critical_failures` | 0 | More than this many `critical` cases fail |
| `max_prompt_growth` | 3.0 | The candidate is longer than this multiple of the submitted prompt |
| `preserve_placeholders` | true | Refinement removed a `{{placeholder}}` |
| `max_avg_latency_ms` | off | Average latency per call is above this |

The approved version is re-evaluated on every run, using its own model and tools, rather than reusing its stored score. This keeps the comparison fair when the dataset or the model's behaviour has changed.

| Result | Registry | Workflow | What to do |
| --- | --- | --- | --- |
| Approved | New version, `approved.yaml` updated | Passes | `git pull` |
| Rejected | New version recorded as `rejected`, `approved.yaml` unchanged | Fails | Fix the prompt or dataset and push again |
| Error (invalid files, API failure) | Nothing recorded | Fails | Fix the cause. The next run retries automatically |

## The registry

The workflow writes this folder, so don't edit it by hand.

```
registry/
  index.json                  Each prompt's approved version and full history
  <name>/approved.yaml        The prompt to use
  <name>/versions/v<N>.yaml   Every evaluated candidate, approved or rejected
  <name>/reports/v<N>.json    Per-case results, gate checks and diff for that version
```

Each version records the refined prompt, your submitted prompt, the model, the tools, scores, the commit and who pushed it.

## Good to know

- **Cost:** each changed prompt makes real API calls. `max_metric_calls` is a soft limit: the sample's setting of 60 made about 140 calls in total in a simulated run, including the 15 for evaluation. Evaluating the approved version adds one call per test case. Lower `max_metric_calls` or choose a cheaper `model` to reduce this.
- **Your prompt may come back unchanged:** the optimizer starts from your prompt and keeps it if nothing scores higher on the train cases. The version's `refined` field shows whether it changed.
- **Tools aren't executed:** the evaluation checks which tool the model chooses and with what arguments, on a single turn. It doesn't run the tools or test multi-step conversations.
- **Results vary between runs:** models aren't deterministic, and the sample has only 15 test cases, so one flaky case moves the pass rate by nearly 7 points. With `max_regression: 0.0`, that alone can reject a candidate. Add test cases for real prompts, then set a small tolerance if needed.
- **Model and tool changes count as changes:** a new model goes through the same comparison as a new prompt.
- **Data leaves the repository:** prompts and dataset cases are sent to your model provider, and model replies are saved in `registry/<name>/reports/`. Don't put real secrets or customer data in them. The code in the sample prompt is made up.

## Run locally

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -r requirements-dev.txt
```

Run the tests, which use a fake model and need no API key:

```bash
.venv/Scripts/python -m pytest tests
```

Run the pipeline with real API calls. It needs your provider's API key set in the environment, and it writes to `registry/`:

```bash
.venv/Scripts/python -m promptops --prompt support-agent
```

On macOS or Linux, use `.venv/bin` instead of `.venv/Scripts`.

The exit code is 0 when everything is approved or skipped, 1 when a candidate is rejected and 2 on an error.
