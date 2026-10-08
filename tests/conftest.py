import pytest
from agent_env.cli import cli
from agent_env.config import reset_config
from click.testing import CliRunner


@pytest.fixture
def state(tmp_path, monkeypatch):
    """A fresh AgentEnv state folder and no config file, so runs and lookups use only this test's local store."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    reset_config()
    yield tmp_path
    reset_config()


@pytest.fixture
def agent_env(state):
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(cli, list(args), catch_exceptions=False)

    return invoke
