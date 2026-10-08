"""A Hub dataset's bundles on disk, and what its card says they need. The HF cache keeps a snapshot as symlinks into
its blob store, and the bundle parser refuses a link that leaves its bundle, so each commit is downloaded into a
plain folder of its own."""

import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import DatasetCard, HfApi
from packaging.requirements import Requirement


@dataclass
class Fetched:
    repo: str
    sha: str
    root: Path
    meta: dict = field(default_factory=dict)

    def bundles(self) -> list[str]:
        folder = self.root / "bundles"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def needs(self, name: str) -> dict:
        return (self.meta.get("bundles") or {}).get(name) or {}


def parse_ref(text: str) -> tuple[str, str | None]:
    repo, _, revision = text.partition("@")
    owner, _, name = repo.partition("/")
    if not owner or not name or "/" in name:
        raise ValueError(f"{text!r}: name the dataset as OWNER/NAME, optionally @REVISION")
    return repo, revision or None


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "")
    return (Path(base) if os.path.isabs(base) else Path.home() / ".cache") / "agentenv-hf"


def fetch(text: str, api: HfApi | None = None) -> Fetched:
    """The dataset at the commit ``text``'s revision names (default: the main branch), its bundles and card."""
    repo, revision = parse_ref(text)
    api = api or HfApi()
    sha = api.repo_info(repo, repo_type="dataset", revision=revision).sha
    root = cache_root() / repo / sha
    api.snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=root,
                          allow_patterns=["README.md", "bundles/**"])
    readme = root / "README.md"
    meta = DatasetCard.load(readme).data.to_dict().get("agentenv") if readme.is_file() else None
    return Fetched(repo, sha, root, meta if isinstance(meta, dict) else {})


def pick(fetched: Fetched, name: str | None) -> str:
    """``name``, else the only bundle, else the one the card names as its default."""
    found = fetched.bundles()
    if not found:
        raise ValueError(f"{fetched.repo} has no bundles/ folder at {fetched.sha[:12]}")
    if name:
        name = name.removeprefix("bundles/").rstrip("/")
        if name not in found:
            raise ValueError(f"{fetched.repo} has no bundle {name!r}; its bundles are {', '.join(found)}")
        return name
    if len(found) == 1:
        return found[0]
    if fetched.meta.get("default") in found:
        return fetched.meta["default"]
    raise ValueError(f"{fetched.repo} holds several bundles; pick one with --bundle: {', '.join(found)}")


def missing_plugins(specs: list[str]) -> list[str]:
    missing = []
    for spec in specs:
        requirement = Requirement(spec)
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing.append(spec)
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            missing.append(spec)
    return missing
