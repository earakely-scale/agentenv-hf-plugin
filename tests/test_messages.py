import json
from pathlib import Path

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
CLAUDE = json.loads((Path(__file__).parent / "fixtures/claude_cli.json").read_text())


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
    claude_without_init = [record for record in CLAUDE if record["type"] != "system"]

    assert detect(spans) is None
    assert to_messages(spans) is None
    assert to_messages("plain text") is None
    assert detect(claude_without_init) is None


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



def test_a_claude_code_stream_is_regrouped_into_responses_and_their_results():
    messages = to_messages(CLAUDE, system="You plan berths.", prompt="Place ship 3.")

    assert detect(CLAUDE) == "claude-cli"
    assert [(m["role"], m.get("tool_call_id")) for m in messages] == [
        ("system", None), ("user", None), ("assistant", None), ("tool", "toolu_a"), ("tool", "toolu_b"),
        ("assistant", None), ("tool", "toolu_c"), ("assistant", None)]
    first = messages[2]
    assert first["content"] == ""
    assert [c["function"] for c in first["tool_calls"]] == [
        {"name": "Read", "arguments": {"path": "plan.md"}},
        {"name": "Agent", "arguments": {"description": "check berths", "prompt": "Check berth 4."}}]
    assert messages[4] == {"role": "tool", "content": "Berth 4 is free from hour 6.", "tool_call_id": "toolu_b",
                           "name": "Agent"}
    assert messages[5]["content"] == "Checking the plan."
    assert messages[6]["content"] == '{"feasible": true}'
    assert messages[7] == {"role": "assistant", "content": "Ship 3 berths at 4 from hour 7."}
    assert "toolu_sub" not in json.dumps(messages)


def test_a_claude_code_stream_without_prompts_starts_at_the_first_response():
    assert to_messages(CLAUDE)[0]["role"] == "assistant"


def test_a_runs_claude_code_turns_are_joined_under_the_system_prompt_its_steps_sent():
    steps = [{"id": "deploy-agent", "type": "deploy_agent", "a2a_agent_id": "claude-code",
              "system_prompt": "You plan berths at <port>."},
             {"id": "ask", "type": "prompt_agent", "prompt": "Place ship 3."},
             {"id": "follow-up", "type": "prompt_agent", "prompt": "And ship 4?"}]
    record = {"instance_id": "v3/week-1", "prompt_responses": [
        {"step_id": "ask", "prompt_text": "Place ship 3.", "agent_name": "default-agent"},
        {"step_id": "follow-up", "prompt_text": "And ship 4?", "agent_name": "default-agent"}],
        "metadata": {"seed": {"port": "Barcelona"}}, "agents": []}
    trajectories = [{"step_id": "ask", "payload": CLAUDE}, {"step_id": "follow-up", "payload": CLAUDE}]

    row = tables.episode_row("week-1", steps, record, trajectories, None)

    assert row["trajectory_format"] == "claude-cli"
    assert row["agent"] == "claude-code"
    assert row["messages"][0] == {"role": "system", "content": "You plan berths at Barcelona."}
    assert [m["content"] for m in row["messages"] if m["role"] == "user"] == ["Place ship 3.", "And ship 4?"]
    assert sum(m["role"] == "system" for m in row["messages"]) == 1
    assert len(row["messages"]) == 1 + 2 * (len(to_messages(CLAUDE)) + 1)
