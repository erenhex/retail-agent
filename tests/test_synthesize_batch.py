"""Tests for batch suite synthesis."""

import json
from pathlib import Path
from unittest import mock

import pytest

from Datapipeline.src.suitegen.run_synthesize import generate_query_and_write
from Datapipeline.src.suitegen.run_synthesize_batch import expand_batch_runs


@pytest.fixture
def combined_base_config(tmp_path):
    cfg = {
        "task": "combined",
        "synthesize_file": "data/suites/synthesize_combined.jsonl",
        "tasks": [
            {"task": "product", "total": 1},
            {"task": "shop", "total": 1},
        ],
    }
    path = tmp_path / "combined.json"
    path.write_text(json.dumps(cfg))
    return path


def test_expand_batch_runs_paths_and_seeds(combined_base_config, tmp_path):
    batch = {
        "base_config": str(combined_base_config.relative_to(tmp_path)),
        "count": 3,
        "seed_start": 1,
        "output_dir": "data/suites",
        "output_template": "combined_30_{i}.jsonl",
    }
    runs = expand_batch_runs(batch, repo_root=tmp_path)

    assert len(runs) == 3
    assert runs[0]["seed"] == 1
    assert runs[2]["seed"] == 3
    assert runs[0]["synthesize_file"] == str(
        tmp_path / "data/suites/combined_30_1.jsonl"
    )
    assert runs[0]["task"] == "combined"


def test_expand_batch_runs_rejects_non_combined_base(tmp_path):
    bad = tmp_path / "product.json"
    bad.write_text(json.dumps({"task": "product", "total": 1}))
    batch = {
        "base_config": "product.json",
        "count": 1,
        "output_dir": "data/suites",
        "output_template": "x_{i}.jsonl",
    }
    with pytest.raises(ValueError, match="combined"):
        expand_batch_runs(batch, repo_root=tmp_path)


def test_generate_query_and_write_includes_category(tmp_path):
    fout_path = tmp_path / "out.jsonl"
    config = {"model_config": {"model": "test", "max_tokens": 10}}

    with open(fout_path, "w") as fout:
        with mock.patch(
            "Datapipeline.src.suitegen.run_synthesize.ask_llm",
            return_value=('', '```json\n{"query": "find shoes"}\n```'),
        ):
            ok = generate_query_and_write(
                "prompt",
                {"product_id": "1"},
                fout,
                config,
                "Shop",
            )

    assert ok
    line = json.loads(fout_path.read_text().strip())
    assert line["category"] == "Shop"
    assert line["query"] == "find shoes"
