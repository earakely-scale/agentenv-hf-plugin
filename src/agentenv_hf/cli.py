import re
from pathlib import Path

import click

from agentenv_hf import dataset, hub, runs, scan
from agentenv_hf.card import card

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@click.group()
def hf():
    """Publish AgentEnv bundles and their recorded runs to the Hugging Face Hub."""


@hf.command()
@click.argument("bundle")
@click.option("--repo", help="The dataset repo to publish to, OWNER/NAME. Created if it doesn't exist.")
@click.option("--name", help="The dataset name of this bundle: its configs, tables and folder. Default: the "
                             "bundle's name.")
@click.option("--runs", "which", type=click.Choice(["latest", "all"]), default="latest", show_default=True,
              help="Each task's newest finished run, or every finished run.")
@click.option("--instance", "instance_ids", multiple=True, help="Publish this run, by instance id. Repeatable; "
                                                               "replaces --runs.")
@click.option("--reward", help="The verifier whose score is the reward. Default: the only verifier, if one.")
@click.option("--split", default="train", show_default=True, help="The split both configs list their table under.")
@click.option("--license", help="The card's license, e.g. apache-2.0, when the card has none.")
@click.option("--card-note", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Markdown for this name's own section of the card, such as its license and data sources. "
                   "Replaces the section a previous publish wrote.")
@click.option("--private", is_flag=True, help="Create the repo private.")
@click.option("--tag", help="Tag the published commit with this revision, e.g. v0.1.0.")
@click.option("--collection", help="Add the dataset to this collection, by slug.")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              help="Write the dataset to this folder instead of pushing it.")
def publish(bundle, repo, name, which, instance_ids, reward, split, license, card_note, private, tag, collection,
            out):
    """Publish BUNDLE (a folder, or an installed bundle's name) and its recorded runs as a Hub dataset.

    The tasks and runs go in two parquet configs, NAME_tasks and NAME_episodes; each run's record and native
    trajectory go in raw/NAME.jsonl and the bundle in bundles/NAME/. A card already on the repo keeps its text;
    the publish adds the rl-environment and agentenv tags, its two configs and, with --card-note, its note.
    Everything goes in one commit, and nothing is published if a file holds a key this machine has set or looks
    like it holds a token.
    """
    if not repo and not out:
        raise click.UsageError("give --repo to publish, or --out to write the dataset to a folder")
    parsed = runs.locate(bundle)
    name = name or parsed.name
    if not NAME.fullmatch(name):
        raise click.BadParameter(f"{name!r}: use letters, digits, '.', '_' and '-'", param_hint="--name")
    built = dataset.build(parsed, name=name, split=split, which=which, instance_ids=instance_ids, reward=reward)
    note = card_note.read_text() if card_note else None
    known = scan.known_values()
    summary = f"{built.tasks} tasks, {built.runs} runs" + (
        f" ({built.unfinished} unfinished left out)" if built.unfinished else "")
    if out:
        files = built.files | {"README.md": card(None, name=name, split=split, repo=repo, description=built.description,
                                                 license=license, note=note).encode()}
        scan.scan(files, known)
        for path, content in files.items():
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_bytes(content)
        click.echo(f"{summary} written to {out}")
        return
    url = hub.push(built, repo, license=license, private=private, tag=tag, collection=collection, known=known,
                   note=note)
    click.echo(f"{summary} published: {url}")
