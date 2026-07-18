import importlib.util
import json
import sys
from pathlib import Path

import pytest

from smart_label import cli as package_cli


def _load_root_main():
    root = Path(__file__).parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("root_cli_entrypoint", root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.mark.parametrize("entrypoint", [package_cli.main, _load_root_main()])
def test_ingest_ranking_cli_errors_are_friendly_in_both_entrypoints(
    tmp_path, monkeypatch, entrypoint
):
    jsonl = tmp_path / "sets.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "set_id": "s1",
                "criterion": {
                    "id": "sentiment",
                    "version": "v1",
                    "prompt": "Which clip is most positive?",
                    "direction": "most",
                },
                "candidates": [{"candidate_id": "a", "path": "a.wav"}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smart-label",
            "ingest-ranking",
            "--jsonl",
            str(jsonl),
            "--db",
            str(tmp_path / "ranking.db"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        entrypoint()

    assert str(error.value) == "line 1: candidates must contain 2 to 8 items"
