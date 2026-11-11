import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_bank
from test_matrix_metrics import make_cases


def test_gram_bank_contains_all_rows_and_correct_subject_holdout(tmp_path):
    source = tmp_path / "source"; make_cases(source)
    folder = tmp_path / "bank"
    joint_bank.prepare(source, folder, degrees=(2,), chunk=51, device="numpy")
    bank = joint_bank.JointBank(source, folder)
    data = bank.load(2)
    assert int(data["n_all"]) == 160
    assert int(data["n_validation"]) == 40
    assert data["g_all"].shape == (28, 28)
    assert np.all(np.isfinite(data["g_all"]))
