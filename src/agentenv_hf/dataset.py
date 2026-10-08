"""A bundle and its runs as dataset files, keyed by the dataset name so several bundles (or versions of one) share a
repo: ``tasks/<name>.parquet``, ``episodes/<name>.parquet``, ``raw/<name>.jsonl`` and ``bundles/<name>/``."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agent_env.bundle.parse import LEAVINGS, Bundle, BundleKind

from agentenv_hf import records, runs, tables


@dataclass
class Dataset:
    name: str
    split: str
    description: str | None
    files: dict[str, bytes]
    tasks: int
    runs: int
    unfinished: int


def build(bundle: Bundle, *, name: str, split: str, which: str, instance_ids: tuple[str, ...] = (),
          reward: str | None = None) -> Dataset:
    rewrite = records.Rewrite(bundle.id_root, name)
    entries = runs.tasks(bundle)
    by_id = {entry.id: entry for entry in entries}
    evals = _evals(bundle, {i: e.name for i, e in by_id.items()})
    task_rows = [rewrite(tables.task_row(e.name, e.config, e.id, evals.get(e.name, []))) for e in entries]
    instances, unfinished = runs.instances(bundle, which, instance_ids)
    episode_rows, raw = [], []
    for instance in instances:
        record = rewrite(records.record(instance))
        trajectories = rewrite(records.trajectories(instance))
        entry = by_id[instance["task_id"]]
        episode_rows.append(tables.episode_row(entry.name, entry.config, record, trajectories, reward))
        raw.append({"episode_id": record["instance_id"], "record": record, "trajectories": trajectories})
    files = {
        f"tasks/{name}.parquet": tables.parquet(task_rows, tables.TASKS),
        f"episodes/{name}.parquet": tables.parquet(episode_rows, tables.EPISODES),
        f"raw/{name}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw).encode(),
        **{f"bundles/{name}/{path}": content for path, content in _bundle_files(bundle.root)},
    }
    return Dataset(name, split, bundle.description, files, len(task_rows), len(episode_rows), unfinished)


def _evals(bundle: Bundle, by_id: dict[str, str]) -> dict[str, list[str]]:
    """Each task's evals, by name. An eval names its tasks by bundle name or by id."""
    names = set(by_id.values())
    member: dict[str, list[str]] = {}
    for entry in bundle.entries:
        if entry.kind is not BundleKind.EVAL:
            continue
        config = entry.config or tomllib.loads(entry.path.read_text())
        for ref in config.get("tasks", []):
            task = ref if ref in names else by_id.get(ref)
            if task:
                member.setdefault(task, []).append(entry.name)
    return member


def _bundle_files(root: Path):
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if path.is_file() and not any(p in LEAVINGS or p == ".git" for p in parts) and not parts[0].startswith("."):
            yield "/".join(parts), path.read_bytes()
