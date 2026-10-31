# Prompt Management

Push a prompt, model name and tool details to a `prompt/*` branch. A workflow refines the prompt with DSPy, evaluates it on a test dataset and compares it with the approved version. If every check passes, it opens a pull request. A person reviews and merges it, then you get the result with `git pull`.

**Only a person can change `main`.** The workflow doesn't run on `main`, never pushes to it and never merges.

```
 you push prompts/<name>/ to prompt/<topic>
          │
          ▼
 Merge main into a working copy, take the registry from main
          │
          ▼
 Changed since the approved version? ── no ──► nothing to do (no API calls)
          │ yes
          ▼
 Refine with DSPy GEPA (train cases only)
          │ candidate prompt
          ▼
 Evaluate on test cases: candidate and currently approved version
          │
          ▼
 Run the gate checks
     │ all pass                        │ any fail
     ▼                                 ▼
 Push candidate/<topic>            Workflow fails, reports kept as an artifact,
 and open a pull request           any earlier pull request is withdrawn
     │
     ▼
 A PERSON reviews and merges into main
     │
     ▼
 You git pull main
```

## Setup

1. Create a GitHub repository and push this folder to `main`. The workflow doesn't run on this push, so it makes no API calls.
2. **Protect `main`.** In **Settings → Rules → Rulesets**, create a branch ruleset targeting the default branch, with an empty bypass list and these rules:
   - Require a pull request before merging, with at least 1 required approval
   - Block force pushes
   - Restrict deletions
3. In **Settings → Actions → General → Workflow permissions**, select **Allow GitHub Actions to create and approve pull requests**. The workflow needs this to open its pull request.
4. In **Settings → Secrets and variables → Actions**, add the key for your model's provider: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`.

Step 2 is what actually enforces the rule. The workflow's own code never touches `main`, but its token has write access to the repository, and only the ruleset stops that token from writing to `main`. GitHub only enforces rulesets on private repositories on paid plans (Pro, Team or Enterprise). On GitHub Free, they're enforced only on public repositories.

## Daily use

1. Create a branch whose name starts with `prompt/`, then edit `prompts/<name>/prompt.yaml` or `dataset.jsonl`, or add a new folder under `prompts/`:

   ```bash
   git switch -c prompt/better-refunds
   ```

2. Commit and push the branch.
3. Check the run on the Actions tab. If the gate passes, a pull request from `candidate/better-refunds` appears, with the check results, failed cases and prompt diff in its description.
4. Review the pull request and merge it. It contains exactly what passed the gate.
5. Pull `main`, then load `registry/<name>/approved.yaml` in your application:

```python
import yaml

approved = yaml.safe_load(open("registry/support-agent/approved.yaml", encoding="utf-8"))
approved["prompt"]  # Still contains {{placeholders}} for you to fill in.
approved["model"], approved["tools"], approved["version"]
```

Pushing again to the same `prompt/*` branch re-evaluates it and updates the same pull request. If the new push is rejected, the pull request is closed, so an outdated candidate can't be merged by mistake.

To re-run without pushing, use **Actions → Prompt pipeline → Run workflow** and select your `prompt/*` branch. Tick `force` to re-evaluate an unchanged prompt.

### If a pull request has conflicts

If someone merges a new version of the same prompt first, your pull request shows conflicts in `registry/`. This is intentional: your candidate was compared with a version that's no longer the approved one. Re-run the workflow on your `prompt/*` branch to evaluate it against the new version and update the pull request.

If your prompt files conflict with `main` as well, the run stops and lists them. Merge `main` into your branch, resolve the conflicts and push.

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

Every check must pass for a pull request to be opened.

| Check | Default | Fails when |
| --- | --- | --- |
| `min_pass_rate` | 0.8 | The candidate's test pass rate is below this |
| `max_regression` | 0.0 | The pass rate drops below the approved version's by more than this |
| `max_critical_failures` | 0 | More than this many `critical` cases fail |
| `max_prompt_growth` | 3.0 | The candidate is longer than this multiple of the submitted prompt |
| `preserve_placeholders` | true | Refinement removed a `{{placeholder}}` |
| `max_avg_latency_ms` | off | Average latency per call is above this |

The approved version is re-evaluated on every run, using its own model and tools, rather than reusing its stored score. This keeps the comparison fair when the dataset or the model's behaviour has changed.

| Result | Pull request | Workflow | What to do |
| --- | --- | --- | --- |
| Passed | Opened or updated | Passes | Review and merge |
| Rejected | None, and any earlier one is closed | Fails | Read the job summary, fix the prompt or dataset and push again |
| Error (invalid files, API failure) | None, and any earlier one is closed | Fails | Fix the cause and push or re-run |
| No changes | None, and any earlier one is closed | Passes | Nothing |

A branch that changes several prompts is all or nothing: if any prompt is rejected, no pull request is opened. Full reports for every run, including rejections, are kept as a workflow artifact named `evaluation-reports`, for as long as your repository retains artifacts (90 days by default).

## The registry

The workflow generates this folder, and it reaches `main` only through a reviewed pull request.

```
registry/
  index.json                  Each prompt's approved version and full history
  <name>/approved.yaml        The prompt to use
  <name>/versions/v<N>.yaml   Every version that passed the gate and was merged
  <name>/reports/v<N>.json    Per-case results, gate checks and diff for that version
```

Each version records the refined prompt, your submitted prompt, the model, the tools, scores, the commit and who pushed it. Merging is what makes a version official, so a candidate whose pull request is closed never appears on `main`.

Registry changes made on a `prompt/*` branch are discarded: each run takes the registry from `main`. This stops a branch from supplying its own `approved.yaml` through the pipeline. It doesn't stop someone opening a hand-written pull request that edits `registry/`, so reviewers should only expect `registry/` changes in pull requests from `candidate/*` branches.

## Security notes

- **Pull request review is the control.** Passing the gate only makes a candidate eligible. A reviewer should read the prompt diff, because the refined prompt is written by a model.
- **Write access means access to secrets.** A push runs the workflow file from the pushed branch, so anyone who can push a `prompt/*` branch can change that copy and read the API keys. Grant write access only to people you'd trust with those keys. The ruleset still stops them changing `main` without a review.
- **Teams should add code owners.** The setting in step 3 also lets workflows approve pull requests, so a changed workflow could approve someone's own pull request. To close this, add a `.github/CODEOWNERS` file naming people as owners of `*`, and turn on **Require review from Code Owners** in the ruleset.
- **Working alone:** GitHub doesn't let you approve your own pull request. Pull requests from this workflow are authored by `github-actions[bot]`, so you can approve those. A pull request you open yourself, such as a code change, needs a second reviewer or a required approval count of 0.
- **Changing the workflow file:** the workflow's token can't push changes to `.github/workflows/`. Change that file in a normal pull request, not a `prompt/*` branch.
- **Data leaves the repository:** prompts and dataset cases are sent to your model provider, and model replies are saved in `registry/<name>/reports/`. Don't put real secrets or customer data in them. The code in the sample prompt is made up.

## Good to know

- **Cost:** each changed prompt makes real API calls. `max_metric_calls` is a soft limit: the sample's setting of 60 made about 140 calls in total in a simulated run, including the 15 for evaluation. Evaluating the approved version adds one call per test case. Every push to a `prompt/*` branch evaluates all prompts that differ from `main` again. Lower `max_metric_calls` or choose a cheaper `model` to reduce this.
- **Your prompt may come back unchanged:** the optimizer starts from your prompt and keeps it if nothing scores higher on the train cases. The version's `refined` field shows whether it changed.
- **Tools aren't executed:** the evaluation checks which tool the model chooses and with what arguments, on a single turn. It doesn't run the tools or test multi-step conversations.
- **Results vary between runs:** models aren't deterministic, and the sample has only 15 test cases, so one flaky case moves the pass rate by nearly 7 points. With `max_regression: 0.0`, that alone can reject a candidate. Add test cases for real prompts, then set a small tolerance if needed.
- **Model and tool changes count as changes:** a new model goes through the same comparison as a new prompt.

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

Run the pipeline with real API calls. It needs your provider's API key set in the environment, and it writes to `registry/`, so discard those changes rather than committing them:

```bash
.venv/Scripts/python -m promptops --prompt support-agent --reports-dir out/reports
```

On macOS or Linux, use `.venv/bin` instead of `.venv/Scripts`.

The exit code is 0 when everything is approved or skipped, 1 when a candidate is rejected and 2 on an error.
