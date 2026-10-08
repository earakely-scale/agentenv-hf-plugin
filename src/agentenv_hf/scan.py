"""The last check before anything leaves the machine: transcripts and bundle files can hold keys no allowlist knows
about, so every file is scanned for token shapes and for the key values this machine has set."""

import io
import json
import os
import re

import pyarrow.parquet as pq
from agent_env.config import get_config
from huggingface_hub import get_token

SHAPES = {
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "API key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{40,}"),
    "Slack token": re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
SECRET_NAMES = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)


def known_values() -> dict[str, str]:
    values = {f"${name}": value for name, value in os.environ.items()
              if SECRET_NAMES.search(name) and len(value) >= 16}
    if token := get_token():
        values["the Hugging Face token"] = token
    try:
        values["the model API key"] = get_config().get_litellm_api_key()
    except Exception:
        pass
    return values


def scan(files: dict[str, bytes], known: dict[str, str]) -> None:
    """Raises ValueError naming the first file that holds a known key or looks like it holds one."""
    for path, content in files.items():
        content = _readable(path, content)
        for label, value in known.items():
            if value.encode() in content:
                raise ValueError(f"{path} holds the value of {label}; nothing was published")
        for label, shape in SHAPES.items():
            if shape.search(content):
                raise ValueError(f"{path} looks like it holds a {label}; nothing was published")


def _readable(path: str, content: bytes) -> bytes:
    """Parquet pages are compressed, so a table is scanned as its rows."""
    if not path.endswith(".parquet"):
        return content
    rows = pq.read_table(io.BytesIO(content)).to_pylist()
    return json.dumps(rows, ensure_ascii=False, default=str).encode()
