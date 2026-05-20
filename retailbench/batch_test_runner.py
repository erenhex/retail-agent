"""Run an agent against multiple problem suites and aggregate scores."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from retailbench.test_runner import _resolve_inference_credentials, run_test


def resolve_suite_paths(suite_glob: str, repo_root: Path | None = None) -> list[Path]:
    """Resolve suite paths from a glob, sorted for stable ordering."""
    root = repo_root or Path(".")
    pattern = suite_glob
    if not Path(suite_glob).is_absolute():
        pattern = str(root / suite_glob)
    paths = sorted(Path(p) for p in glob.glob(pattern))
    return [p for p in paths if p.is_file()]


def run_batch(
    agent_file: str,
    suite_paths: list[Path],
    *,
    max_workers: int = 3,
    timeout: int = 1800,
    skip_reasoning: bool = False,
    seed_start: int | None = None,
) -> dict:
    """Run the agent on each suite; return summary dict."""
    suites_out = []
    scores = []

    for idx, suite_path in enumerate(suite_paths):
        seed = None
        if seed_start is not None:
            seed = seed_start + idx

        print(f"\n{'═' * 60}")
        print(f"Suite {idx + 1}/{len(suite_paths)}: {suite_path}")
        if seed is not None:
            print(f"Expected seed: {seed}")
        print(f"{'═' * 60}\n")

        score = run_test(
            agent_file,
            str(suite_path),
            max_workers=max_workers,
            timeout=timeout,
            skip_reasoning=skip_reasoning,
        )

        entry = {
            "file": str(suite_path),
            "score": score,
            "status": "ok" if score >= 0 else "failed",
        }
        if seed is not None:
            entry["seed"] = seed
        suites_out.append(entry)
        if score >= 0:
            scores.append(score)

    summary = {
        "agent": agent_file,
        "suites": suites_out,
    }
    if scores:
        summary["mean_score"] = sum(scores) / len(scores)
        summary["min_score"] = min(scores)
        summary["max_score"] = max(scores)
    else:
        summary["mean_score"] = None
        summary["min_score"] = None
        summary["max_score"] = None

    return summary


def print_summary_table(summary: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"{'Suite':<40} {'Score':>8}  {'Status'}")
    print(f"{'─' * 60}")
    for entry in summary["suites"]:
        name = Path(entry["file"]).name
        score = entry["score"]
        score_str = f"{score:.4f}" if score >= 0 else "FAILED"
        print(f"{name:<40} {score_str:>8}  {entry['status']}")
    print(f"{'─' * 60}")
    if summary.get("mean_score") is not None:
        print(
            f"Mean: {summary['mean_score']:.4f}  "
            f"Min: {summary['min_score']:.4f}  "
            f"Max: {summary['max_score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="batch_test_runner",
        description="Test an agent against multiple benchmark suites",
    )
    parser.add_argument("--agent-file", required=True, help="Agent Python file to test")
    parser.add_argument(
        "--suite-glob",
        required=True,
        help="Glob for suite files (e.g. data/suites/combined_30_*.jsonl)",
    )
    parser.add_argument(
        "--results-file",
        default="logs/batch_results.json",
        help="Where to write JSON summary (default: logs/batch_results.json)",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip reasoning quality scoring",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional seed label for suite 1 (increments per file); for summary metadata",
    )

    args = parser.parse_args()

    api_key, _, _ = _resolve_inference_credentials()
    if not api_key:
        print(
            "Error: no OpenRouter API key set.\n"
            "  Set OPENROUTER_API_KEY in your shell or .env.",
            file=sys.stderr,
        )
        sys.exit(2)

    suite_paths = resolve_suite_paths(args.suite_glob)
    if not suite_paths:
        print(f"Error: no suites matched: {args.suite_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Agent:  {args.agent_file}")
    print(f"Suites: {len(suite_paths)} matched")

    summary = run_batch(
        args.agent_file,
        suite_paths,
        max_workers=args.max_workers,
        timeout=args.timeout,
        skip_reasoning=args.skip_reasoning,
        seed_start=args.seed_start,
    )

    print_summary_table(summary)

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"\nWrote summary to {results_path}")

    failed = sum(1 for s in summary["suites"] if s["status"] != "ok")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
