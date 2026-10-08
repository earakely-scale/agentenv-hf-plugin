import json

import datasets

from agentenv_hf import tables
from agentenv_hf.messages import detect, to_messages

FLAT = {"model": "m", "messages": [
    {"role": "system", "content": "Plan the week."},
    {"role": "user", "content": [{"type": "text", "text": "Ships: "}, {"type": "text", "text": "3"}]},
    {"role": "assistant", "content": "", "reasoning": "", "tool_calls": [
        {"id": "c1", "name": "check_plan", "arguments": '{"plan": [{"ship": 0, "berth_hour": 4}]}'}]},
    {"role": "tool", "tool_call_id": "c1", "name": "check_plan", "content": '{"feasible": true}'},
    {"role": "assistant", "content": "done"},
]}
NESTED = [{"role": "assistant", "content": None, "tool_calls": [
    {"id": "c2", "type": "function", "function": {"name": "advance", "arguments": "{}"}}]}]


def test_flat_tool_calls_become_openai_calls_with_object_arguments():
    messages = to_messages(FLAT)

    assert detect(FLAT) == "chat"
    assert messages[1] == {"role": "user", "content": "Ships: 3"}
    assert messages[2]["tool_calls"] == [{"id": "c1", "type": "function", "function": {
        "name": "check_plan", "arguments": {"plan": [{"ship": 0, "berth_hour": 4}]}}}]
    assert messages[3] == {"role": "tool", "content": '{"feasible": true}', "tool_call_id": "c1", "name": "check_plan"}


def test_nested_calls_and_a_bare_message_list():
    assert to_messages(NESTED) == [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "advance", "arguments": {}}}]}]


def test_other_formats_are_left_alone():
    spans = [{"name": "invoke_agent", "attributes": {"gen_ai.operation.name": "invoke_agent"}}]

    assert detect(spans) is None
    assert to_messages(spans) is None
    assert to_messages("plain text") is None


def test_messages_round_trip_through_parquet_as_objects(tmp_path):
    row = {"episode_id": "e", "messages": to_messages(FLAT), "scores": {"v": 0.5}}
    (tmp_path / "e.parquet").write_bytes(tables.parquet([row], tables.EPISODES))

    loaded = datasets.load_dataset("parquet", data_files=str(tmp_path / "e.parquet"), split="train")[0]

    assert loaded["scores"] == {"v": 0.5}
    call = loaded["messages"][2]["tool_calls"][0]
    assert call["function"]["arguments"] == {"plan": [{"ship": 0, "berth_hour": 4}]}
    assert json.loads(json.dumps(loaded["messages"]))[3]["tool_call_id"] == "c1"


def test_reward_is_the_only_score_or_the_named_one():
    found = tables.scores({"a": {"score": 0.25}, "check": {"passed": True}, "flag": {"score": True}})

    assert found == {"a": 0.25}
    assert tables.reward(found, None) == 0.25
    assert tables.reward({"a": 0.25, "b": 1.0}, None) is None
    assert tables.reward({"a": 0.25, "b": 1.0}, "b") == 1.0
