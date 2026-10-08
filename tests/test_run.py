import json
import shutil
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_env.store.routing import namespace_routing
from agent_env.task.store import task_instances

from agentenv_hf import fetch

HELLO = Path(str(files("agent_env") / "examples" / "hello"))
SHA = "0123456789abcdef0123456789abcdef01234567"
CARD = """---
tags:
- rl-environment
- agentenv
{meta}---

# A test dataset
"""


class FakeHub:
    """Serves ``repo`` from a folder: ``repo_info`` resolves every revision to SHA, ``snapshot_download`` copies it."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.downloads = 0

    def repo_info(self, repo_id, repo_type, revision=None):
        return SimpleNamespace(sha=SHA)

    def snapshot_download(self, repo_id, repo_type, revision, local_dir, allow_patterns):
        assert revision == SHA and repo_type == "dataset"
        self.downloads += 1
        shutil.copytree(self.repo, local_dir, dirs_exist_ok=True)
        return str(local_dir)


@pytest.fixture
def hub(state, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))
    repo = state / "repo"
    shutil.copytree(HELLO, repo / "bundles" / "hello")
    (repo / "README.md").write_text(CARD.format(meta=""))
    fake = FakeHub(repo)
    monkeypatch.setattr(fetch, "HfApi", lambda: fake)
    return fake


def _runs() -> list:
    with namespace_routing():
        return task_instances("@local/hf/me/ds/hello/hello")


def _card(hub: FakeHub, agentenv: dict) -> None:
    (hub.repo / "README.md").write_text(CARD.format(meta=f"agentenv: {json.dumps(agentenv)}\n"))


def test_a_hub_bundle_runs_under_a_fixed_id_root_and_a_fresh_download_writes_nothing_new(agent_env, hub, state):
    first = agent_env("hf", "run", "me/ds", "--yes")
    shutil.rmtree(state / "cache")
    second = agent_env("hf", "run", "me/ds@v1", "--yes")

    assert first.exit_code == 0, first.output
    assert f"me/ds at {SHA[:12]}, bundle hello" in first.output
    assert "instance @local/hf/me/ds/hello/hello-" in first.output
    assert hub.downloads == 2
    assert "tasks/hello.json v1: passed" in second.output
    assert [r.task_version for r in _runs()] == [1, 1]


def test_a_missing_plugin_stops_the_run_and_names_the_command(agent_env, hub):
    _card(hub, {"bundles": {"hello": {"plugins": ["agentenv-not-installed>=1"]}}})

    result = agent_env("hf", "run", "me/ds", "--yes")

    assert result.exit_code != 0
    assert "agent-env plugin add 'agentenv-not-installed>=1'" in result.output
    assert _runs() == []


def test_a_bundle_that_needs_setup_names_the_cards_setup_command(agent_env, hub):
    steps = [{"id": "deploy", "type": "deploy_env", "env_id": "an-env-only-setup-registers"}]
    (hub.repo / "bundles" / "hello" / "tasks" / "hello.json").write_text(json.dumps(steps))
    _card(hub, {"bundles": {"hello": {"setup": "agent-env demo setup"}}})

    result = agent_env("hf", "run", "me/ds", "--yes")

    assert result.exit_code != 0
    assert "the dataset card sets bundle hello up with: agent-env demo setup" in result.output


def test_without_a_terminal_it_runs_only_with_yes(agent_env, hub):
    result = agent_env("hf", "run", "me/ds")

    assert result.exit_code != 0
    assert "pass --yes" in result.output
    assert _runs() == []


def test_a_dry_run_shows_the_plan_and_runs_nothing(agent_env, hub):
    result = agent_env("hf", "run", "me/ds", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "Would run:\n  tasks/hello.json v1" in result.output
    assert _runs() == []


def test_several_bundles_need_a_choice_or_the_cards_default(agent_env, hub):
    shutil.copytree(HELLO, hub.repo / "bundles" / "greet")

    unchosen = agent_env("hf", "run", "me/ds", "--dry-run")
    _card(hub, {"default": "greet"})
    by_default = agent_env("hf", "run", "me/ds", "--dry-run")
    named = agent_env("hf", "run", "me/ds", "--bundle", "bundles/hello", "--dry-run")

    assert "pick one with --bundle: greet, hello" in unchosen.output
    assert "bundle greet" in by_default.output
    assert "bundle hello" in named.output


def test_a_dataset_is_named_as_owner_and_name():
    assert fetch.parse_ref("me/ds@v0.3.1") == ("me/ds", "v0.3.1")
    assert fetch.parse_ref("me/ds") == ("me/ds", None)
    for bad in ("ds", "me/ds/extra", "/ds"):
        with pytest.raises(ValueError, match="OWNER/NAME"):
            fetch.parse_ref(bad)


def test_installed_plugins_are_found_by_name_and_version():
    assert fetch.missing_plugins(["click", "click>=8"]) == []
    assert fetch.missing_plugins(["click>=999", "agentenv-not-installed"]) == ["click>=999", "agentenv-not-installed"]
