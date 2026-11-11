"""Exercise real runner shapes, finite differences, and strict output freezing."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_finite
from test_matrix_metrics import make_cases


def test_shared_bank_finite_pipeline_never_reads_forbidden_inputs(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    # Use a shared numeric X bank, detected from W rather than model name.
    x = np.array([[-1., 0.], [0., -1.], [1., 0.], [0., 1.], [1., 1.]])
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path)
        w[:, 4:] = np.tile(x, (32, 1))
        np.save(path, w)
    out = tmp_path / "out"
    rows = run_finite.analyze(source, out, estimators=("poly3",), centers=8)
    assert rows
    assert json.loads((out / "protocol.json").read_text())["status"] == "COMPLETE"
    assert json.loads((out / "response_quality.json").read_text())["mode"] == "numerically verified shared observed X"
    audit = json.loads((out / "read_audit.json").read_text())
    assert audit["all_subjects"] == 32
    assert audit["all_rows"] == 160
    assert audit["forbidden_files_read"] == []
    assert not any("R_best" in f or "Zstd" in f for f in audit["files"])
    assert (out / "common_support_subjects.npy").exists()
