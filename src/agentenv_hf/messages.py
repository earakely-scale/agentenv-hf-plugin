"""A trajectory as chat messages in the shape TRL and transformers read: OpenAI roles and tool calls, with each call's
arguments an object rather than the JSON string the OpenAI API sends."""

import json
from typing import Any

CHAT = "chat"
CLAUDE_CLI = "claude-cli"


def detect(payload: Any) -> str | None:
    if _chat(payload) is not None:
        return CHAT
    if _claude_cli(payload):
        return CLAUDE_CLI
    return None


def to_messages(payload: Any, *, system: str | None = None, prompt: str | None = None) -> list[dict] | None:
    """The trajectory's messages, or None for a format this plugin doesn't read yet. ``system`` and ``prompt`` are
    what the run sent the agent, which a Claude Code stream doesn't record."""
    messages = _chat(payload)
    if messages is not None:
        return [_message(m) for m in messages]
    if _claude_cli(payload):
        head = [{"role": "system", "content": system}] if system else []
        head += [{"role": "user", "content": prompt}] if prompt else []
        return head + _claude_cli_messages(payload)
    return None


def _chat(payload: Any) -> list[dict] | None:
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if isinstance(messages, list) and messages and all(isinstance(m, dict) and "role" in m for m in messages):
        return messages
    return None


def _claude_cli(payload: Any) -> bool:
    """``claude -p --output-format stream-json``: a list of records that starts each session with system/init."""
    return isinstance(payload, list) and any(
        isinstance(r, dict) and r.get("type") == "system" and r.get("subtype") == "init" for r in payload)


def _claude_cli_messages(records: list[dict]) -> list[dict]:
    """Claude Code streams one model response as several records sharing a message id, with the results of its
    earlier tool calls in between, so each response is regrouped and followed by its calls' results. Sub-agents'
    records are left out; their work reaches the main thread as the result of the call that started them."""
    responses: dict[str, dict] = {}
    order: list[tuple[str, str]] = []
    results: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("parent_tool_use_id") is not None:
            continue
        message = record.get("message") or {}
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        if record.get("type") == "assistant":
            key = message.get("id") or record.get("uuid") or str(len(order))
            if key not in responses:
                responses[key] = {"text": [], "calls": []}
                order.append(("assistant", key))
            for block in blocks:
                if block.get("type") == "text" and block.get("text"):
                    responses[key]["text"].append(block["text"])
                elif block.get("type") == "tool_use":
                    responses[key]["calls"].append({"id": block.get("id"), "type": "function", "function": {
                        "name": block.get("name"), "arguments": block.get("input") or {}}})
        elif record.get("type") == "user":
            texts = []
            for block in blocks:
                if block.get("type") == "tool_result":
                    results[block.get("tool_use_id")] = _result_text(block.get("content"))
                elif block.get("type") == "text" and block.get("text"):
                    texts.append(block["text"])
            if texts:
                order.append(("user", "".join(texts)))
    messages = []
    for kind, value in order:
        if kind == "user":
            messages.append({"role": "user", "content": value})
            continue
        response = responses[value]
        if not response["text"] and not response["calls"]:
            continue
        message = {"role": "assistant", "content": "\n\n".join(response["text"])}
        if response["calls"]:
            message["tool_calls"] = response["calls"]
        messages.append(message)
        messages += [{"role": "tool", "content": results[call["id"]], "tool_call_id": call["id"],
                      "name": call["function"]["name"]} for call in response["calls"] if call["id"] in results]
    return messages


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) and part.get("type") == "text"
                       else json.dumps(part, ensure_ascii=False) for part in content)
    return _text(content)


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
