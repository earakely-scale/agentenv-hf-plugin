"""A run's record as it may be published. ``to_safe_dict`` strips only a few keys, so the record is rebuilt from an
allowlist: deployment URLs, cards and env state never leave, and the bundle's local id root and the home folder are
rewritten so no local path does."""

import json
from pathlib import Path
from typing import Any

from agent_env.config import get_config

INSTANCE_KEYS = ("instance_id", "task_id", "task_version", "status", "error", "created_at_utc", "completed_at_utc")
CONTEXT_KEYS = ("agent_model", "default_agent_model", "agent_artifact_id", "agent_harness")
METADATA_KEYS = ("run_group_id", "seed", "verifications", "failed_steps")
RESPONSE_KEYS = ("step_id", "prompt_id", "agent_name", "model", "prompt_text", "response", "tool_call_count",
                 "structured_output", "error_type", "error_code", "error_message")
ENV_KEYS = ("env_id", "env_version", "env_provider_type")
AGENT_KEYS = ("agent_name", "agent_id", "sandbox_type")


class Rewrite:
    """Replaces the bundle's id root with the dataset name, and the home folder with ``~``, in every string."""

    def __init__(self, id_root: str, name: str):
        self.pairs = [(id_root, name), (str(Path.home()), "~")]

    def text(self, value: str) -> str:
        for old, new in self.pairs:
            value = value.replace(old, new)
        return value

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k): self(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        return value


def _pick(source: dict, keys: tuple[str, ...]) -> dict:
    return {key: source[key] for key in keys if source.get(key) is not None}


def record(instance: dict) -> dict:
    context = instance.get("context") or {}
    metadata = context.get("metadata") or {}
    return _pick(instance, INSTANCE_KEYS) | {
        "completed_steps": [_pick(step, ("step_id", "status")) for step in instance.get("completed_steps") or []],
        **_pick(context, CONTEXT_KEYS),
        "envs": [_pick(env, ENV_KEYS) for env in context.get("deployed_envs") or []],
        "agents": [_pick(agent, AGENT_KEYS) for agent in context.get("deployed_agents") or []],
        "prompt_responses": [_pick(response, RESPONSE_KEYS) for response in context.get("prompt_responses") or []],
        "metadata": _pick(metadata, METADATA_KEYS),
    }


def trajectories(instance: dict) -> list[dict]:
    """Each prompt step's native trajectory, parsed when it is JSON. The agents' formats differ and none is tagged."""
    found = []
    for response in (instance.get("context") or {}).get("prompt_responses") or []:
        url = response.get("agent_trajectory_s3_uri") or response.get("agent_trajectory_object_url")
        if not url:
            continue
        raw = get_config().get_object_store_at(url).get(url)
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw.decode("utf-8", errors="replace")
        if isinstance(payload, dict) and set(payload) == {"trajectory"}:
            payload = payload["trajectory"]
        found.append({"step_id": response.get("step_id"), "payload": payload})
    return found
