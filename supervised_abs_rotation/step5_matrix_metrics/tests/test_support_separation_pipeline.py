import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_finite
import run_tail_support
import run_separation
from test_matrix_metrics import make_cases


def test_support_and_separation_use_verified_fits_and_preserve_endpoints(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    x = np.arange(10).reshape(5, 2)
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path); w[:, 4:] = np.tile(x, (32, 1)); np.save(path, w)
    fields = tmp_path / "fields"
    run_finite.analyze(source, fields, estimators=("poly3", "rbf1"), centers=8)
    for name, runner in (("support", run_tail_support), ("separation", run_separation)):
        output = tmp_path / name
        kwargs = dict(anchors_count=2) if name == "separation" else {}
        rows = runner.analyze(source, fields, output, **kwargs)
        assert rows
        assert max(r["endpoint_error"] for r in rows) < 1e-6
        assert json.loads((output / "protocol.json").read_text())["status"] == "COMPLETE"
        audit = json.loads((output / "read_audit.json").read_text())
        assert audit["forbidden_files_read"] == []
        assert audit["all_subjects"] == 32
