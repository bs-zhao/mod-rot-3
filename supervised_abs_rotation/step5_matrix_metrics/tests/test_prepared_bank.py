import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_response_bank
import fitted_bank
import run_separation
from test_matrix_metrics import make_cases


def test_immutable_prepared_bank_is_not_mislabeled_as_case_scoring(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    x = np.arange(10).reshape(5, 2)
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path); w[:, 4:] = np.tile(x, (32, 1)); np.save(path, w)
    folder = tmp_path / "bank"
    prepare_response_bank.prepare(source, folder, estimators=("poly3", "rbf1"), centers=8)
    protocol = json.loads((folder / "protocol.json").read_text())
    assert protocol["status"] == "COMPLETE"
    assert "no case scores" in protocol["artifact_type"]
    assert not (folder / "blind_scores.csv").exists()
    bank = fitted_bank.FittedBank(source, folder)
    assert bank.model("poly3")["grams"][0].shape == (35, 35)
    assert bank.model("rbf1")["grams"][0].shape == (8, 8)
    rows = run_separation.analyze(source, folder, tmp_path / "scored", anchors_count=2)
    assert rows
    # A changed prepared model must never be silently reused.
    with (folder / "model_poly3.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        fitted_bank.FittedBank(source, folder)
