"""The bundle a dataset is built from and its recorded runs. AgentEnv keeps no record of a bundle or eval run, only
each run's task instance, so runs are found by task id: one query per task of the bundle."""

import os
from pathlib import Path

from agent_env.bundle.installed import find_bundle
from agent_env.bundle.parse import Bundle, BundleEntry, BundleKind, parse_bundle
from agent_env.config import get_config
from agent_env.store import Filter


def locate(text: str) -> Bundle:
    """The bundle ``text`` names, read as ``agent-env run`` reads it: a folder, or an installed bundle's name."""
    if text.startswith((".", "/", "~")) or os.path.isdir(text):
        return parse_bundle(Path(text))
    found = find_bundle(text)
    return parse_bundle(found.root, id_root=found.id_root)


def tasks(bundle: Bundle) -> list[BundleEntry]:
    return [entry for entry in bundle.entries if entry.kind is BundleKind.TASK]


def instances(bundle: Bundle, which: str, ids: tuple[str, ...] = ()) -> tuple[list[dict], int]:
    """The runs to publish and how many unfinished ones were left out. ``which`` is ``latest`` (each task's newest
    finished run) or ``all``; ``ids`` picks runs by instance id instead."""
    store = get_config().get_document_store()
    known = {entry.id for entry in tasks(bundle)}
    if ids:
        found = []
        for instance_id in ids:
            doc = store.find_one("task_instances", Filter.of(instance_id=instance_id))
            if doc is None or doc["task_id"] not in known:
                raise ValueError(f"{instance_id}: no recorded run of a task in {bundle.name}")
            found.append(doc)
        return found, 0
    picked, unfinished = [], 0
    for entry in tasks(bundle):
        docs = sorted(store.query("task_instances", Filter.of(task_id=entry.id)),
                      key=lambda doc: (doc.get("created_at_utc") or "", doc.get("completed_at_utc") or ""))
        finished = [doc for doc in docs if doc["status"] != "running"]
        unfinished += len(docs) - len(finished)
        if which == "latest":
            finished = finished[-1:]
        picked += finished
    return picked, unfinished
