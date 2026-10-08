"""The push. Following NeMo Gym's upload, the card is read at the commit the push builds on and every file goes in
one commit pinned to that parent, so a concurrent push fails instead of interleaving. ``upload_folder`` can split
an upload into several commits, so it isn't used."""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import EntryNotFoundError

from agentenv_hf.card import card
from agentenv_hf.dataset import Dataset
from agentenv_hf.scan import scan


def push(dataset: Dataset, repo: str, *, license: str | None, private: bool, tag: str | None,
         collection: str | None, known: dict[str, str], api: HfApi | None = None) -> str:
    scan(dataset.files, known)
    api = api or HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    parent = api.repo_info(repo, repo_type="dataset").sha
    try:
        existing = Path(api.hf_hub_download(repo, "README.md", repo_type="dataset", revision=parent)).read_text()
    except EntryNotFoundError:
        existing = None
    readme = card(existing, name=dataset.name, split=dataset.split, repo=repo, description=dataset.description,
                  license=license).encode()
    scan({"README.md": readme}, known)
    files = dataset.files | {"README.md": readme}
    stale = [path for path in api.list_repo_files(repo, repo_type="dataset", revision=parent)
             if path.startswith(f"bundles/{dataset.name}/") and path not in files]
    operations = [CommitOperationAdd(path, content) for path, content in files.items()]
    operations += [CommitOperationDelete(path) for path in stale]
    commit = api.create_commit(repo, operations, repo_type="dataset", parent_commit=parent,
                               commit_message=f"Publish {dataset.name}: {dataset.tasks} tasks, {dataset.runs} runs")
    if tag:
        api.create_tag(repo, tag=tag, repo_type="dataset", revision=commit.oid)
    if collection:
        api.add_collection_item(collection, item_id=repo, item_type="dataset", exists_ok=True)
    return commit.commit_url
