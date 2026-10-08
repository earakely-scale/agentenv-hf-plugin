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


def test_a_names_note_is_replaced_on_republish_and_other_names_keep_theirs():
    first = card(None, name="v3", split="eval", repo="me/ds", description=None, license=None,
                 note="## v3 data\n\nCC BY-SA 4.0.\n")
    both = card(first, name="hello", split="eval", repo="me/ds", description=None, license=None,
                note="## hello\n\nApache-2.0.")
    replaced = card(both, name="v3", split="eval", repo="me/ds", description=None, license=None,
                    note="## v3 data\n\nCC BY-SA 4.0, Port de Barcelona.")
    unchanged = card(replaced, name="v3", split="eval", repo="me/ds", description=None, license=None)

    assert replaced.count("<!-- agentenv-hf:v3 -->") == 1
    assert "CC BY-SA 4.0, Port de Barcelona." in replaced and "CC BY-SA 4.0.\n" not in replaced
    assert replaced.index("<!-- agentenv-hf:v3 -->") < replaced.index("<!-- agentenv-hf:hello -->")
    assert "Apache-2.0." in replaced
    assert unchanged == replaced


def test_a_note_goes_after_a_hand_written_cards_text():
    text = card(EXISTING, name="v3", split="eval", repo="me/ds", description=None, license=None, note="Data: CC BY-SA.")

    assert text.endswith("Its own text.\n\n<!-- agentenv-hf:v3 -->\nData: CC BY-SA.\n<!-- /agentenv-hf:v3 -->\n")


def test_a_bundles_needs_go_in_the_agentenv_table_beside_the_others():
    v3 = {"plugins": ["agentenv-portsim @ git+https://example.org/p@v1"], "setup": "agent-env portsim setup"}
    first = card(None, name="v3", split="eval", repo="me/ds", description=None, license=None, needs=v3)
    both = card(first, name="v2", split="eval", repo="me/ds", description=None, license=None,
                needs={"plugins": ["agentenv-portsim"]})
    unchanged = card(both, name="v3", split="eval", repo="me/ds", description=None, license=None)

    assert _meta(both)["agentenv"] == {"bundles": {"v3": v3, "v2": {"plugins": ["agentenv-portsim"]}}}
    assert _meta(unchanged)["agentenv"] == _meta(both)["agentenv"]
