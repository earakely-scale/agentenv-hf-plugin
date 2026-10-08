from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import EntryNotFoundError

from agentenv_hf import tables
from agentenv_hf.dataset import Dataset
from agentenv_hf.hub import push


class FakeApi:
    def __init__(self, card_path=None):
        self.calls = []
        self.card_path = card_path

    def create_repo(self, repo, **kwargs):
        self.calls.append(("create_repo", repo, kwargs))

    def repo_info(self, repo, **kwargs):
        return SimpleNamespace(sha="parent-sha")

    def hf_hub_download(self, repo, filename, **kwargs):
        if self.card_path is None:
            raise EntryNotFoundError("no README.md")
        return self.card_path

    def list_repo_files(self, repo, **kwargs):
        return ["README.md", "bundles/hello/old.json", "bundles/other/keep.json", "bundles/hello/tasks/hello.json"]

    def create_commit(self, repo, operations, **kwargs):
        self.calls.append(("create_commit", operations, kwargs))
        return SimpleNamespace(oid="new-sha", commit_url="https://huggingface.co/datasets/me/ds/commit/new-sha")

    def create_tag(self, repo, **kwargs):
        self.calls.append(("create_tag", kwargs))

    def add_collection_item(self, slug, **kwargs):
        self.calls.append(("add_collection_item", slug, kwargs))


def _dataset(**files):
    base = {"tasks/hello.parquet": tables.parquet([{"task": "hello"}], tables.TASKS),
            "episodes/hello.parquet": tables.parquet([{"episode_id": "hello/hello-1"}], tables.EPISODES),
            "raw/hello.jsonl": b"", "bundles/hello/tasks/hello.json": b"[]"}
    return Dataset("hello", "train", None, base | files, 1, 1, 0)


def test_one_commit_on_the_parent_it_read_then_the_tag_and_collection():
    api = FakeApi()

    url = push(_dataset(), "me/ds", license=None, private=True, tag="v0.1.0", collection="me/col-123", known={},
               api=api)

    assert url.endswith("new-sha")
    names = [call[0] for call in api.calls]
    assert names == ["create_repo", "create_commit", "create_tag", "add_collection_item"]
    assert api.calls[0][2] == {"repo_type": "dataset", "private": True, "exist_ok": True}
    _, operations, kwargs = api.calls[1]
    assert kwargs["parent_commit"] == "parent-sha"
    added = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
    deleted = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}
    assert added == {"README.md", "tasks/hello.parquet", "episodes/hello.parquet", "raw/hello.jsonl",
                     "bundles/hello/tasks/hello.json"}
    assert deleted == {"bundles/hello/old.json"}
    assert api.calls[2][1]["revision"] == "new-sha"


def test_an_existing_card_is_merged_not_replaced(tmp_path):
    existing = tmp_path / "README.md"
    existing.write_text("---\ntags:\n- simulation\n---\n\n# Mine\n")
    api = FakeApi(card_path=existing)

    push(_dataset(), "me/ds", license=None, private=False, tag=None, collection=None, known={}, api=api)

    readme = next(op for op in api.calls[1][1] if op.path_in_repo == "README.md").path_or_fileobj.decode()
    assert readme.endswith("# Mine\n")
    assert "- simulation\n- rl-environment\n- agentenv\n" in readme


def test_a_known_key_stops_before_the_repo_is_created():
    api = FakeApi()

    with pytest.raises(ValueError, match="raw/hello.jsonl holds the value of the model API key"):
        push(_dataset(**{"raw/hello.jsonl": b'{"key": "abc-secret-123"}'}), "me/ds", license=None, private=False,
             tag=None, collection=None, known={"the model API key": "abc-secret-123"}, api=api)

    assert api.calls == []


def test_a_token_inside_a_compressed_table_is_found():
    leaky = tables.parquet([{"episode_id": "e", "response": "use hf_" + "Zz9" * 12}], tables.EPISODES)

    with pytest.raises(ValueError, match="episodes/hello.parquet looks like it holds a Hugging Face token"):
        push(_dataset(**{"episodes/hello.parquet": leaky}), "me/ds", license=None, private=False, tag=None,
             collection=None, known={}, api=FakeApi())
