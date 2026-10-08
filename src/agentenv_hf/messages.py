"""A trajectory as chat messages in the shape TRL and transformers read: OpenAI roles and tool calls, with each call's
arguments an object rather than the JSON string the OpenAI API sends."""

import json
from typing import Any

CHAT = "chat"


def detect(payload: Any) -> str | None:
    if _chat(payload) is not None:
        return CHAT
    return None


def to_messages(payload: Any) -> list[dict] | None:
    """The trajectory's messages, or None for a format this plugin doesn't read yet."""
    messages = _chat(payload)
    if messages is None:
        return None
    return [_message(m) for m in messages]


def _chat(payload: Any) -> list[dict] | None:
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if isinstance(messages, list) and messages and all(isinstance(m, dict) and "role" in m for m in messages):
        return messages
    return None


def _message(message: dict) -> dict:
    out = {"role": message["role"], "content": _text(message.get("content"))}
    if message.get("tool_calls"):
        out["tool_calls"] = [_tool_call(call) for call in message["tool_calls"]]
    for key in ("tool_call_id", "name"):
        if message.get(key) is not None:
            out[key] = message[key]
    return out


def _tool_call(call: dict) -> dict:
    function = call.get("function") or call
    return {"id": call.get("id"), "type": "function",
            "function": {"name": function.get("name"), "arguments": _arguments(function.get("arguments"))}}


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value) if value.strip() else {}
        except ValueError:
            return value
    return {} if value is None else value


def _text(content: Any) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return json.dumps(content, ensure_ascii=False)
