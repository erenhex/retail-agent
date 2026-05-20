"""Generate multiple combined suites with distinct seeds."""

import os
import sys
from pathlib import Path

import ujson as json

from .run_synthesize import (
    _REPO_ROOT,
    _configure_randomness,
    synthesize_combined,
)
from .util.env import load_repo_env

os.chdir(_REPO_ROOT)
load_repo_env(_REPO_ROOT)


def expand_batch_runs(batch_config: dict, repo_root: Path | None = None) -> list[dict]:
    """Build per-run combined configs from a batch spec."""
    root = repo_root or _REPO_ROOT
    base_path = root / batch_config["base_config"]
    with open(base_path, "r") as fin:
        base = json.load(fin)

    if base.get("task") != "combined":
        raise ValueError(
            f"base_config task must be 'combined', got {base.get('task')!r}"
        )

    count = int(batch_config["count"])
    seed_start = int(batch_config.get("seed_start", 1))
    output_dir = root / batch_config.get("output_dir", "data/suites")
    template = batch_config.get("output_template", "combined_{i}.jsonl")

    runs = []
    for i in range(1, count + 1):
        run = dict(base)
        run["seed"] = seed_start + i - 1
        run["synthesize_file"] = str(output_dir / template.format(i=i))
        runs.append(run)
    return runs


def run_batch(batch_config: dict, *, continue_on_error: bool = False) -> None:
    runs = expand_batch_runs(batch_config)
    total = len(runs)
    for idx, run_config in enumerate(runs, start=1):
        out = run_config["synthesize_file"]
        seed = run_config["seed"]
        print(f"[{idx}/{total}] seed={seed} -> {out}")
        _configure_randomness(run_config)
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            synthesize_combined(run_config)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if not continue_on_error:
                raise
            print("Continuing after error (--continue-on-error).", file=sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m Datapipeline.src.suitegen.run_synthesize_batch "
            "<batch_config.json> [--continue-on-error]",
            file=sys.stderr,
        )
        sys.exit(2)

    config_file = sys.argv[1]
    continue_on_error = "--continue-on-error" in sys.argv[2:]

    with open(config_file, "r") as fin:
        batch_config = json.load(fin)

    run_batch(batch_config, continue_on_error=continue_on_error)


if __name__ == "__main__":
    main()
