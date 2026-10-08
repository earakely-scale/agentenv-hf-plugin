import re
import sys
from pathlib import Path

import click
from agent_env.bundle import BundleError, RunInterrupted, dry_run_bundle, run_bundle
from agent_env.cli.run import _report_dry_run, _report_teardown, _summarize
from agent_env.task.interrupts import Interrupts

from agentenv_hf import dataset, fetch, hub, runs, scan
from agentenv_hf.card import card

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@click.group()
def hf():
    """Publish AgentEnv bundles and their recorded runs to the Hugging Face Hub, and run bundles from it."""


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
@click.option("--requires", multiple=True, metavar="SPEC",
              help="A plugin this bundle needs, as `agent-env plugin add` takes it. Repeatable; written to the card, "
                   "where `agent-env hf run` checks it.")
@click.option("--setup", metavar="COMMAND", help="The command that sets the bundle up after its plugins are added, "
                                                 "such as one that builds and registers its images.")
@click.option("--private", is_flag=True, help="Create the repo private.")
@click.option("--tag", help="Tag the published commit with this revision, e.g. v0.1.0.")
@click.option("--collection", help="Add the dataset to this collection, by slug.")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              help="Write the dataset to this folder instead of pushing it.")
def publish(bundle, repo, name, which, instance_ids, reward, split, license, card_note, requires, setup, private, tag,
            collection, out):
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
    needs = {key: value for key, value in (("plugins", list(requires)), ("setup", setup)) if value} or None
    known = scan.known_values()
    summary = f"{built.tasks} tasks, {built.runs} runs" + (
        f" ({built.unfinished} unfinished left out)" if built.unfinished else "")
    if out:
        files = built.files | {"README.md": card(None, name=name, split=split, repo=repo, description=built.description,
                                                 license=license, note=note, needs=needs).encode()}
        scan.scan(files, known)
        for path, content in files.items():
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_bytes(content)
        click.echo(f"{summary} written to {out}")
        return
    url = hub.push(built, repo, license=license, private=private, tag=tag, collection=collection, known=known,
                   note=note, needs=needs)
    click.echo(f"{summary} published: {url}")


@hf.command("run")
@click.argument("dataset_ref", metavar="DATASET")
@click.option("--bundle", "bundle_name", help="The bundle to run, by its folder under bundles/. Default: the only one, "
                                              "or the one the card names as its default.")
@click.option("--task", "tasks", multiple=True, help="Run this task, by name or id. Repeatable.")
@click.option("--eval", "evals", multiple=True, help="Run this eval's tasks, by name or id. Repeatable.")
@click.option("--model", default=None, help="The model the agent runs on. The judge keeps its own.")
@click.option("--sandbox", default=None,
              help="The sandbox provider to deploy on; comma-separated for a fallback chain.")
@click.option("--dry-run", is_flag=True, help="Download and check the bundle, show what would run, and run nothing.")
@click.option("--yes", "-y", is_flag=True, help="Run without asking first.")
@click.pass_context
def run(ctx, dataset_ref, bundle_name, tasks, evals, model, sandbox, dry_run, yes):
    """Run a bundle from a Hub dataset: DATASET is OWNER/NAME, optionally @REVISION (a branch, tag or commit).

    The dataset is downloaded at that commit, and its bundle runs as `agent-env run` runs a folder, with ids rooted
    at @local/hf/OWNER/NAME/BUNDLE, so a run of the same commit reuses what an earlier run wrote. Plugins the card
    says the bundle needs must be installed first; nothing the card names is installed or run for you. A bundle can
    build images and run commands on this machine, so the run shows what it would do and asks before it starts.
    """
    fetched = fetch.fetch(dataset_ref)
    name = fetch.pick(fetched, bundle_name)
    needs = fetched.needs(name)
    missing = fetch.missing_plugins(needs.get("plugins") or [])
    if missing:
        adds = "\n".join(f"  agent-env plugin add '{spec}'" for spec in missing)
        raise click.ClickException(f"bundle {name} needs plugins that aren't installed; add them with:\n{adds}")
    root, id_root = fetched.root / "bundles" / name, f"@local/hf/{fetched.repo}/{name}"
    click.echo(f"{fetched.repo} at {fetched.sha[:12]}, bundle {name}")
    try:
        _report_dry_run(dry_run_bundle(root, tasks=tasks, evals=evals, sandbox=sandbox, on_progress=click.echo,
                                       id_root=id_root))
    except BundleError as e:
        if needs.get("setup"):
            e.add_note(f"the dataset card sets bundle {name} up with: {needs['setup']}")
        raise
    if dry_run:
        return
    if not yes:
        if not sys.stdin.isatty():
            raise click.ClickException("a bundle from the Hub runs only after you confirm; pass --yes to run it here")
        click.confirm(f"Run bundle {name} from {fetched.repo}? It builds and runs the code above on this machine",
                      abort=True)
    verbose = ctx.find_root().params.get("verbose")
    try:
        result = run_bundle(root, tasks=tasks, evals=evals, model=model, sandbox=sandbox, on_progress=click.echo,
                            id_root=id_root)
    except RunInterrupted as stop:
        with Interrupts():
            _summarize(stop.result, verbose)
            _report_teardown(stop.result)
        raise SystemExit(128 + stop.signum) from None
    with Interrupts():
        _summarize(result, verbose)
        _report_teardown(result)
    if result.failed:
        raise SystemExit(1)
