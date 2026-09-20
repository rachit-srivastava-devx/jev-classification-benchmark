"""The corpus is committed gzipped, so every loading path has to be exercised.

A silent fallback that picked the wrong file, or a missing-file error naming a
path that was never going to exist, would both waste a reader's time in exactly
the moment they are trying to reproduce the run.
"""

from __future__ import annotations

import gzip
import json

import pytest

from jevdemo.rag.corpus import load_tasks

PAYLOAD = {"depth": 100, "tasks": [{"query_id": "q1"}]}


def _write_plain(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(PAYLOAD))
    return p


def _write_packed(tmp_path):
    p = tmp_path / "tasks.json.gz"
    p.write_bytes(gzip.compress(json.dumps(PAYLOAD).encode("utf-8")))
    return p


def test_a_plain_json_corpus_loads(tmp_path):
    assert load_tasks(_write_plain(tmp_path)) == PAYLOAD


def test_a_gzipped_corpus_loads_when_named_directly(tmp_path):
    assert load_tasks(_write_packed(tmp_path)) == PAYLOAD


def test_naming_the_plain_file_falls_back_to_the_gzipped_one(tmp_path):
    """The committed artifact is the .gz, but every command line and default in
    the repo is written against the .json name."""
    _write_packed(tmp_path)
    assert load_tasks(tmp_path / "tasks.json") == PAYLOAD


def test_the_plain_file_wins_when_both_are_present(tmp_path):
    """A freshly rebuilt corpus sits next to a stale .gz until it is repacked.
    Preferring the uncompressed one means a rebuild takes effect immediately."""
    _write_packed(tmp_path)
    fresh = tmp_path / "tasks.json"
    fresh.write_text(json.dumps({"depth": 20, "tasks": []}))
    assert load_tasks(fresh)["depth"] == 20


def test_a_missing_corpus_names_both_paths_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        load_tasks(tmp_path / "tasks.json")
    assert "tasks.json" in str(err.value)
    assert "tasks.json.gz" in str(err.value)
