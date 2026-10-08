"""The dataset's tables. Dicts of no fixed shape (scores, verifier output, tool-call arguments) are Arrow JSON columns,
which ``datasets`` reads back as Python objects, so TRL gets tool-call arguments as objects."""

import io
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from agent_env.task_step.task_step import TaskStep

from agentenv_hf.messages import detect, to_messages

JSON = pa.json_()
MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("tool_calls", pa.list_(JSON)),
                     ("tool_call_id", pa.string()), ("name", pa.string())])

TASKS = pa.schema([
    ("task", pa.string()), ("task_id", pa.string()), ("evals", pa.list_(pa.string())), ("prompt", pa.string()),
    ("envs", pa.list_(pa.string())), ("agents", pa.list_(pa.string())), ("verifiers", pa.list_(pa.string())),
    ("num_steps", pa.int64()), ("steps", JSON),
])

EPISODES = pa.schema([
    ("episode_id", pa.string()), ("task", pa.string()), ("task_version", pa.int64()), ("run_group", pa.string()),
    ("status", pa.string()), ("created_utc", pa.string()), ("completed_utc", pa.string()), ("model", pa.string()),
    ("agent", pa.string()), ("reward", pa.float64()), ("scores", JSON), ("prompt", pa.string()),
    ("response", pa.string()), ("messages", pa.list_(MESSAGE)), ("tool_calls", pa.int64()),
    ("trajectory_format", pa.string()), ("failed_step", pa.string()), ("error_type", pa.string()),
    ("error", pa.string()), ("verifications", JSON), ("structured_output", JSON),
])


def task_row(name: str, steps: list[dict], task_id: str, evals: list[str]) -> dict:
    prompt = next((s["prompt"] for s in steps if s.get("type") == "prompt_agent" and isinstance(s.get("prompt"), str)),
                  None)
    return {"task": name, "task_id": task_id, "evals": evals, "prompt": prompt,
            "envs": _ids(steps, "deploy_env", "env_id"), "agents": _ids(steps, "deploy_agent", "a2a_agent_id"),
            "verifiers": [s["verifier_id"] for s in steps if s.get("verifier_id")], "num_steps": len(steps),
            "steps": steps}


def scores(verifications: dict) -> dict[str, float]:
    return {vid: float(entry["score"]) for vid, entry in verifications.items()
            if isinstance(entry, dict) and isinstance(entry.get("score"), int | float)
            and not isinstance(entry["score"], bool)}


def reward(found: dict[str, float], verifier: str | None) -> float | None:
    """The named verifier's score, or the only one; with several and none named, no single reward is right."""
    if verifier:
        return found.get(verifier)
    return next(iter(found.values())) if len(found) == 1 else None


def episode_row(task: str, steps: list[dict], record: dict, trajectories: list[dict], verifier: str | None) -> dict:
    responses = record["prompt_responses"]
    metadata = record["metadata"]
    failed = metadata.get("failed_steps") or []
    found = scores(metadata.get("verifications") or {})
    payload = trajectories[-1]["payload"] if trajectories else None
    last = responses[-1] if responses else {}
    error = next((r for r in reversed(responses) if r.get("error_message")), None)
    return {
        "episode_id": record["instance_id"], "task": task, "task_version": record.get("task_version"),
        "run_group": metadata.get("run_group_id"), "status": record.get("status"),
        "created_utc": record.get("created_at_utc"), "completed_utc": record.get("completed_at_utc"),
        "model": last.get("model") or record.get("agent_model"),
        "agent": record.get("agent_artifact_id") or _agent(steps, last.get("agent_name")),
        "reward": reward(found, verifier), "scores": found,
        "prompt": next((r["prompt_text"] for r in responses if r.get("prompt_text")), None),
        "response": last.get("response"),
        "messages": None if payload is None else to_messages(payload),
        "tool_calls": sum(r["tool_call_count"] for r in responses if "tool_call_count" in r) if responses else None,
        "trajectory_format": None if payload is None else detect(payload) or "unknown",
        "failed_step": failed[-1].get("step_id") if failed else None,
        "error_type": (error or {}).get("error_type") or (failed[-1].get("error_type") if failed else None),
        "error": (error or {}).get("error_message") or (failed[-1].get("error") if failed else None)
        or record.get("error"),
        "verifications": metadata.get("verifications"), "structured_output": last.get("structured_output"),
    }


def parquet(rows: list[dict], schema: pa.Schema) -> bytes:
    plain = pa.schema([(f.name, _storage(f.type)) for f in schema])
    table = pa.Table.from_pylist([{f.name: _encode(row.get(f.name), f.type) for f in schema} for row in rows],
                                 schema=plain).cast(schema)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _agent(steps: list[dict], name: str | None) -> str | None:
    """The agent a run's prompt went to. The run records only the name it deployed it under, so the id comes from
    the task's deploy_agent step; a step without one deployed the configured default agent."""
    name = name or TaskStep.DEFAULT_AGENT_NAME
    return next((s.get("a2a_agent_id") for s in steps if s.get("type") == "deploy_agent"
                 and (s.get("agent_name") or TaskStep.DEFAULT_AGENT_NAME) == name), None)


def _ids(steps: list[dict], step_type: str, key: str) -> list[str]:
    return list(dict.fromkeys(s[key] for s in steps if s.get("type") == step_type and s.get(key)))


def _storage(t: pa.DataType) -> pa.DataType:
    if isinstance(t, pa.BaseExtensionType):
        return t.storage_type
    if pa.types.is_list(t):
        return pa.list_(_storage(t.value_type))
    if pa.types.is_struct(t):
        return pa.struct([(f.name, _storage(f.type)) for f in t])
    return t


def _encode(value: Any, t: pa.DataType) -> Any:
    if value is None:
        return None
    if isinstance(t, pa.BaseExtensionType):
        return json.dumps(value, ensure_ascii=False)
    if pa.types.is_list(t):
        return [_encode(v, t.value_type) for v in value]
    if pa.types.is_struct(t):
        return {f.name: _encode(value.get(f.name), f.type) for f in t}
    return value
