import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_finite
import run_profiled
from test_matrix_metrics import make_cases


def test_profiled_pipeline_checks_source_and_reuses_only_derived_fields(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    x = np.arange(10).reshape(5, 2)
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path); w[:, 4:] = np.tile(x, (32, 1)); np.save(path, w)
    fields = tmp_path / "fields"
    run_finite.analyze(source, fields, estimators=("poly3",), centers=8)
    output = tmp_path / "out"
    rows = run_profiled.analyze(source, fields, output)
    assert rows
    audit = json.loads((output / "read_audit.json").read_text())
    assert audit["all_subjects"] == 32
    assert audit["forbidden_files_read"] == []
    assert "field_poly3.npz" in audit["field_sha256"]
    assert json.loads((output / "protocol.json").read_text())["status"] == "COMPLETE"
