"""Tests for batch_test_runner."""

from pathlib import Path
from unittest import mock

import pytest

from retailbench.batch_test_runner import resolve_suite_paths, run_batch


def test_resolve_suite_paths_sorted(tmp_path):
    (tmp_path / "combined_30_2.jsonl").write_text('{"query": "b"}\n')
    (tmp_path / "combined_30_1.jsonl").write_text('{"query": "a"}\n')
    (tmp_path / "other.jsonl").write_text("{}\n")

    paths = resolve_suite_paths("combined_30_*.jsonl", repo_root=tmp_path)
    names = [p.name for p in paths]
    assert names == ["combined_30_1.jsonl", "combined_30_2.jsonl"]


def test_run_batch_summary_aggregation(tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s2 = tmp_path / "s2.jsonl"
    s1.write_text("{}\n")
    s2.write_text("{}\n")

    with mock.patch(
        "retailbench.batch_test_runner.run_test",
        side_effect=[0.5, 0.7, -1.0],
    ):
        summary = run_batch(
            "agent.py",
            [s1, s2, tmp_path / "s3.jsonl"],
            skip_reasoning=True,
            seed_start=1,
        )

    assert len(summary["suites"]) == 3
    assert summary["suites"][0]["seed"] == 1
    assert summary["suites"][0]["score"] == 0.5
    assert summary["suites"][2]["status"] == "failed"
    assert summary["mean_score"] == pytest.approx(0.6)
    assert summary["min_score"] == 0.5
    assert summary["max_score"] == 0.7
