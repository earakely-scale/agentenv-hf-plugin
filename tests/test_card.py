import yaml

from agentenv_hf.card import card

EXISTING = """---
license: cc-by-sa-4.0
tags:
- agentenv
- simulation
configs:
- config_name: v2_tasks
  data_files:
  - split: eval
    path: tasks/v2.parquet
  default: true
- config_name: v3_tasks
  data_files:
  - split: eval
    path: tasks/old.parquet
---

# Hand-written card

Its own text.
"""


def _meta(text: str) -> dict:
    return yaml.safe_load(text.split("---")[1])


def test_a_new_card_lists_both_configs_and_the_tags():
    meta = _meta(card(None, name="hello", split="train", repo="me/ds", description="Say hi.", license="apache-2.0"))

    assert meta["tags"] == ["rl-environment", "agentenv"]
    assert meta["license"] == "apache-2.0"
    assert meta["configs"][0] == {"config_name": "hello_tasks", "default": True,
                                  "data_files": [{"split": "train", "path": "tasks/hello.parquet"}]}
    assert meta["configs"][1]["data_files"][0]["path"] == "episodes/hello.parquet"


def test_an_existing_card_keeps_its_text_license_default_and_other_configs():
    text = card(EXISTING, name="v3", split="eval", repo="me/ds", description=None, license="apache-2.0")
    meta = _meta(text)

    assert text.endswith("# Hand-written card\n\nIts own text.\n")
    assert meta["license"] == "cc-by-sa-4.0"
    assert meta["tags"] == ["agentenv", "simulation", "rl-environment"]
    assert [(c["config_name"], c.get("default")) for c in meta["configs"]] == [
        ("v2_tasks", True), ("v3_tasks", None), ("v3_episodes", None)]
    assert meta["configs"][1]["data_files"][0]["path"] == "tasks/v3.parquet"
