import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_finite
import run_anova
import run_anova_support
from test_matrix_metrics import make_cases


def test_exact_anova_pipeline_preserves_rotation_control(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    x = np.arange(10).reshape(5, 2)
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path); w[:, 4:] = np.tile(x, (32, 1)); np.save(path, w)
    fields = tmp_path / "fields"
    run_finite.analyze(source, fields, estimators=("poly3", "rbf1"), centers=8)
    output = tmp_path / "out"
    rows = run_anova.analyze(source, fields, output, anchors_count=2)
    assert rows
    assert max(r["control_relative_drift"] for r in rows) < 1e-6
    assert all(r["numerical_adequacy"] for r in rows)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["status"] == "COMPLETE"
    assert protocol["empirical_input_independence_assumed"] is False
    support = tmp_path / "support"
    rows = run_anova_support.analyze(output, support)
    assert rows
    assert all("entropy" not in r["metric"] for r in rows)
    assert json.loads((support / "protocol.json").read_text())["status"] == "COMPLETE"
