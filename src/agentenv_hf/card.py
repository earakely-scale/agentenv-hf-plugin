"""The dataset card. An existing card keeps its text and its other configs; the publish adds the tags that list the
dataset under the Hub's RL environments and this name's two configs. All configs are parquet, because the Hub reads
every config of a repo with the builder the first one picks. A name's note (its license and sources) sits between
markers the publish owns, so republishing replaces it and leaves the other names' notes alone."""

from huggingface_hub import DatasetCard

TAGS = ("rl-environment", "agentenv")


def configs(name: str, split: str) -> list[dict]:
    return [{"config_name": f"{name}_{table}", "data_files": [{"split": split, "path": f"{table}/{name}.parquet"}]}
            for table in ("tasks", "episodes")]


def card(existing: str | None, *, name: str, split: str, repo: str | None, description: str | None,
         license: str | None, note: str | None = None, needs: dict | None = None) -> str:
    result = DatasetCard(existing) if existing else DatasetCard(_body(name, repo, description))
    data = result.data
    data.tags = list(dict.fromkeys([*(data.get("tags") or []), *TAGS]))
    ours = configs(name, split)
    kept = [c for c in data.get("configs") or [] if c.get("config_name") not in {c["config_name"] for c in ours}]
    if not any(c.get("default") for c in kept):
        ours[0]["default"] = True
    data.configs = kept + ours
    if license and not data.get("license"):
        data.license = license
    if needs:
        table = dict(data.get("agentenv") or {})
        table["bundles"] = {**(table.get("bundles") or {}), name: needs}
        data.agentenv = table
    if note is not None:
        result.text = _with_note(result.text, name, note)
    return str(result)


def _with_note(text: str, name: str, note: str) -> str:
    start, end = f"<!-- agentenv-hf:{name} -->", f"<!-- /agentenv-hf:{name} -->"
    block = f"{start}\n{note.strip()}\n{end}"
    before, found, rest = text.partition(start)
    if found and end in rest:
        return before + block + rest.split(end, 1)[1]
    return f"{text.rstrip()}\n\n{block}\n"


def _body(name: str, repo: str | None, description: str | None) -> str:
    download = f"hf download {repo or '<owner>/<name>'} --repo-type dataset --local-dir agentenv-dataset"
    return f"""---
tags: {list(TAGS)}
---

# {name}

{description or "An AgentEnv bundle and its recorded runs."}

An [AgentEnv](https://www.agentenvframework.com) bundle with its recorded runs, published with
[agentenv-hf](https://github.com/earakely-scale/agentenv-hf-plugin).

## Tables

- `{name}_tasks`: one row per task of the bundle, with its prompt, envs, verifiers and steps.
- `{name}_episodes`: one row per recorded run, with its status, model, reward and scores, the prompt and response,
  and the agent's transcript as chat `messages` when its trajectory format is one agentenv-hf reads.

`raw/{name}.jsonl` holds each run's record and native trajectory, and `bundles/{name}/` the bundle itself.

## Run it

```bash
pip install agentenv-framework huggingface_hub
{download}
agent-env run agentenv-dataset/bundles/{name}
```
"""
