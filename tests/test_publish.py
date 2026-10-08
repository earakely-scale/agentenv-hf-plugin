import shutil
from importlib.resources import files
from pathlib import Path

import datasets
import yaml

HELLO = Path(str(files("agent_env") / "examples" / "hello"))


def _card(out: Path) -> dict:
    return yaml.safe_load((out / "README.md").read_text().split("---")[1])


def test_hello_publishes_its_tasks_and_latest_run(agent_env, state):
    assert agent_env("run", "hello").exit_code == 0
    assert agent_env("run", "hello").exit_code == 0

    result = agent_env("hf", "publish", "hello", "--out", "ds")

    assert result.exit_code == 0, result.output
    assert "1 tasks, 1 runs" in result.output
    out = state / "ds"
    card = _card(out)
    assert {"rl-environment", "agentenv"} <= set(card["tags"])
    assert [c["config_name"] for c in card["configs"]] == ["hello_tasks", "hello_episodes"]
    tasks = datasets.load_dataset(str(out), "hello_tasks", split="train")
    assert tasks[0]["task_id"] == "hello/hello"
    assert tasks[0]["verifiers"] == ["hello"]
    episode = datasets.load_dataset(str(out), "hello_episodes", split="train")[0]
    assert episode["episode_id"].startswith("hello/hello-")
    assert (episode["status"], episode["reward"], episode["scores"]) == ("completed", 1.0, {"hello": 1.0})
    assert (out / "bundles/hello/tasks/hello.json").read_bytes() == (HELLO / "tasks/hello.json").read_bytes()
    assert (out / "raw/hello.jsonl").read_text().count("\n") == 1


def test_a_card_note_goes_in_the_written_card(agent_env, state):
    (state / "note.md").write_text("## hello data\n\nApache-2.0, from agentenv-framework.\n")

    result = agent_env("hf", "publish", "hello", "--card-note", "note.md", "--out", "ds")

    assert result.exit_code == 0, result.output
    readme = (state / "ds/README.md").read_text()
    assert "<!-- agentenv-hf:hello -->\n## hello data\n\nApache-2.0, from agentenv-framework.\n" in readme


def test_all_runs_and_no_local_paths(agent_env, state):
    bundle = shutil.copytree(HELLO, state / "my-hello")
    assert agent_env("run", str(bundle)).exit_code == 0
    assert agent_env("run", str(bundle)).exit_code == 0

    result = agent_env("hf", "publish", str(bundle), "--runs", "all", "--name", "greet", "--out", "ds")

    assert "1 tasks, 2 runs" in result.output
    for path in (state / "ds").rglob("*"):
        if path.is_file() and path.suffix != ".parquet":
            text = path.read_text()
            assert str(state) not in text and "@local" not in text, path
    episodes = datasets.load_dataset(str(state / "ds"), "greet_episodes", split="train")
    assert all(e.startswith("greet/hello-") for e in episodes["episode_id"])


def test_a_token_in_the_bundle_stops_the_publish(agent_env, state):
    bundle = shutil.copytree(HELLO, state / "leaky")
    (bundle / "artifacts/greeting/notes.txt").write_text("token hf_" + "a1B2c3D4e5" * 4 + "\n")

    result = agent_env("hf", "publish", str(bundle), "--out", "ds")

    assert result.exit_code != 0
    assert "bundles/leaky/artifacts/greeting/notes.txt" in result.output
    assert "nothing was published" in result.output
    assert not (state / "ds").exists()


def test_an_instance_of_another_bundle_is_refused(agent_env, state):
    result = agent_env("hf", "publish", "hello", "--instance", "elsewhere-12345678", "--out", "ds")

    assert result.exit_code != 0
    assert "no recorded run of a task in hello" in result.output
