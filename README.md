# agentenv-hf

Publish [AgentEnv](https://www.agentenvframework.com) bundles and their recorded runs to the
[Hugging Face Hub](https://huggingface.co/datasets) as datasets, and run bundles straight from them.

`agent-env hf publish` takes a bundle you have run with `agent-env run` and writes a dataset with the bundle's tasks,
one row per recorded run (status, model, reward, scores and the agent's transcript as chat messages), each run's
record and native trajectory, and the bundle folder itself. The card is tagged `rl-environment` and `agentenv`, so
the dataset is listed under the Hub's [RL environments](https://huggingface.co/datasets?other=rl-environment).
`agent-env hf run OWNER/NAME` downloads such a dataset at a pinned commit and runs one of its bundles.

## Install

In an agent-env install:

```bash
agent-env plugin add 'agentenv-hf @ git+https://github.com/earakely-scale/agentenv-hf-plugin@v0.3.0'
```

Or install both together:

```bash
uv tool install agentenv-framework --with 'agentenv-hf @ git+https://github.com/earakely-scale/agentenv-hf-plugin@v0.3.0'
```

Pushing needs a Hugging Face token with write access: log in with `hf auth login` (or
`uvx --from huggingface_hub hf auth login`), or set `HF_TOKEN`.

## Publish

```bash
agent-env run hello                                           # the built-in bundle: no Docker, model or config
agent-env hf publish hello --out hello-dataset                 # write the dataset to a folder and look first
agent-env hf publish hello --repo you/hello-agentenv --private # then push it
```

`BUNDLE` is what `agent-env run` takes: a folder, or the name of a bundle an installed package provides. The runs
are found in the stores `agent-env run` wrote them to.

| Option | |
|---|---|
| `--repo OWNER/NAME` | The dataset repo. Created if it doesn't exist (`--private` creates it private). |
| `--out DIR` | Write the dataset to a folder instead of pushing it. |
| `--name NAME` | The dataset name of this bundle: its configs, tables and folder. Default: the bundle's name. |
| `--runs latest\|all` | Each task's newest finished run (the default), or every finished run. |
| `--instance ID` | Publish this run instead, by instance id. Repeatable. |
| `--reward VERIFIER` | The verifier whose score is `reward`. By default it's the score of a run's only verifier, and empty when there are several. |
| `--split NAME` | The split both configs list their table under. Default `train`. |
| `--license ID` | The card's license (e.g. `apache-2.0`), when the card has none. |
| `--card-note FILE` | Markdown for this name's own section of the card, such as its license and data sources. A republish replaces the section; without the option, the section stays. |
| `--requires SPEC` | A plugin the bundle needs, as `agent-env plugin add` takes it (`name @ git+https://...@v1`). Repeatable. Written to the card for `agent-env hf run` to check. |
| `--setup COMMAND` | The command that sets the bundle up once its plugins are added, such as one that builds and registers its images. Written to the card. |
| `--tag REV` | Tag the published commit, e.g. `v0.1.0`, so others can pin it. |
| `--collection SLUG` | Add the dataset to a collection. |

Several bundles, or versions of one, can share a repo: each publish writes only its own name's files, configs and
note, and a card already on the repo keeps its text and its other configs and notes. Everything goes in one commit on top of the commit
the publish read the card from.

## Run a bundle from the Hub

```bash
agent-env hf run you/hello-agentenv --bundle hello          # asks before it runs
agent-env hf run you/hello-agentenv@v0.1.0 --bundle hello --dry-run
```

`DATASET` is `OWNER/NAME`, optionally `@REVISION` (a branch, tag or commit). The run:

1. Resolves the revision to its commit, and downloads the card and `bundles/` into
   `~/.cache/agentenv-hf/OWNER/NAME/COMMIT` (plain files, since bundle folders can't be read through the Hub cache's
   symlinks).
2. Picks the bundle: `--bundle`, else the only one, else the card's `default`.
3. Checks the plugins the card says the bundle needs are installed, and stops with the `agent-env plugin add`
   commands if they aren't. It never installs or runs anything the card names.
4. Makes the checks `agent-env run` makes before its first task and shows what it would build and run. On a problem
   such as an env that isn't registered yet, it prints the card's setup command.
5. Asks before it runs, since a bundle can build images and run commands on this machine. `--yes` skips the
   question; without a terminal, it runs only with `--yes`.
6. Runs the bundle as `agent-env run` runs a folder, with `--task`, `--eval`, `--model` and `--sandbox` as there. Ids
   are rooted at `@local/hf/OWNER/NAME/BUNDLE`, wherever the download lands, so a run of the same commit reuses what
   an earlier one wrote, and runs can be published again with `agent-env hf publish`.

What a bundle needs lives in the card, under an `agentenv` table that `hf publish --requires --setup` writes:

```yaml
agentenv:
  default: dock-v1-eval
  bundles:
    dock-v1-eval:
      plugins:
        - agentenv-portsim @ git+https://github.com/earakely-scale/agentenv-portsim-plugin@v0.3.1
      setup: agent-env portsim setup --agent
```

## What's in the dataset

| Path | Config | |
|---|---|---|
| `tasks/NAME.parquet` | `NAME_tasks` | One row per task of the bundle |
| `episodes/NAME.parquet` | `NAME_episodes` | One row per published run |
| `raw/NAME.jsonl` | | Each run's record and its native trajectory, one run per line |
| `bundles/NAME/` | | The bundle folder, as it is on disk |
| `README.md` | | The card |

```python
from datasets import load_dataset

episodes = load_dataset("you/hello-agentenv", "hello_episodes", split="train")
```

**`NAME_tasks`:** `task`, `task_id`, `evals` (the bundle's evals that include it), `prompt`, `envs`, `agents`,
`verifiers`, `num_steps` and `steps` (the task's steps, as an object).

**`NAME_episodes`:**

| Column | |
|---|---|
| `episode_id`, `task`, `task_version` | The run, as `NAME/<task>-<suffix>`, and the task it ran |
| `run_group` | Runs started by one `agent-env run` share it |
| `status`, `created_utc`, `completed_utc` | `completed`, `failed` or `cancelled`; the times have minute resolution |
| `model`, `agent` | The model the agent ran on, and the agent the task deployed |
| `reward`, `scores` | See `--reward`; `scores` maps each verifier to its score |
| `prompt`, `response` | The first prompt sent to the agent, and its last response |
| `messages` | The agent's transcript as chat messages, when its trajectory format is one this plugin reads |
| `tool_calls` | How many tool calls the agent reported |
| `trajectory_format` | `chat`, `claude-cli`, `unknown`, or empty for a run with no agent |
| `failed_step`, `error_type`, `error` | Why a run failed |
| `verifications`, `structured_output` | The verifiers' full output, and the agent's structured output |

`messages` follows the chat shape TRL's `SFTTrainer` and transformers chat templates read: OpenAI roles, `tool_calls`
of `{id, type: "function", function: {name, arguments}}` with `arguments` an object, and `tool_call_id` and `name` on
tool messages. The dicts with no fixed shape (`scores`, `verifications`, `structured_output`, `steps` and the tool
calls) are Arrow JSON columns, which `datasets` 5.1 (the version the Hub's viewer runs) reads back as Python
objects.

Trajectory formats read so far:

- **Chat-message records** (`chat`): a `messages` list of role dicts with OpenAI or flat `{id, name, arguments}` tool
  calls.
- **Claude Code** (`claude-cli`): the `claude -p --output-format stream-json` event stream of an agent that wraps
  Claude Code. The stream splits a model response into several records, with tool results between them, so each
  response is regrouped and followed by its calls' results. It records neither the system prompt nor the user prompt:
  the system prompt is the one the task's `prompt_agent` or `deploy_agent` step sent (filled from the run's seed), and
  each prompt step's turn starts with the prompt it sent. Sub-agents' own records are left out; their work reaches the
  conversation as the result of the call that started them. Thinking is left out.

A run in another format keeps its trajectory in `raw/NAME.jsonl` and leaves `messages` empty.

## What is kept out

- **Local paths:** the bundle's local ids (`@local/~/...`) are rewritten to `NAME`, and the home folder to `~`.
- **Deployment details:** a run's record is rebuilt from a list of the fields worth publishing, so endpoints, agent
  and environment cards, tunnel URLs and env state are left out.
- **Keys:** before anything is written or pushed, every file is scanned for the values of this machine's model key,
  Hugging Face token and `*KEY*`/`*TOKEN*`/`*SECRET*`/`*PASSWORD*` environment variables, and for token shapes
  (Hugging Face, OpenAI-style, AWS, GitHub, Slack, private keys). A hit stops the publish and names the file.

Transcripts are published as the agent wrote them; read a run's `messages` before you make a dataset public.

## Compatibility

agent-env keeps no record of a bundle or eval run, so the plugin finds runs by task id, through the framework's public
read API for recorded runs (`agent_env.task.store.task_instances` and `find_task_instance`, agentenv-framework
0.9.1298 and later). It still reads bundles with agent-env's own parser, trajectories through the object store, and runs bundles with
`run_bundle` and `agent-env run`'s own reporting, which are framework internals rather than the plugin surface, so
CI runs against the framework's latest release and its `main`.

## Develop

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest && .venv/bin/ruff check .
```

The tests run the built-in `hello` bundle into a temporary state folder; they need no network, Docker or model.

## License

Apache-2.0.
