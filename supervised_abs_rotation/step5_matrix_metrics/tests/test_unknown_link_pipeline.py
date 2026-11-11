import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_unknown_link
from test_matrix_metrics import make_cases


def test_unknown_link_pipeline_freezes_scores_before_labels(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    x = np.array([[-1., 0.], [0., -1.], [1., 0.], [0., 1.], [1., 1.]])
    for path in (source / "cases").glob("*/W.npy"):
        w = np.load(path); w[:, 4:] = np.tile(x, (32, 1)); np.save(path, w)
    out = tmp_path / "out"
    rows = run_unknown_link.analyze(source, out, families=("poly3",), iterations=1)
    assert rows
    protocol = json.loads((out / "protocol.json").read_text())
    reveal = json.loads((out / "reveal.json").read_text())
    assert protocol["status"] == "COMPLETE"
    assert protocol["scores_sha256"] == reveal["scores_sha256"]
    audit = json.loads((out / "read_audit.json").read_text())
    assert audit["all_subjects"] == 32
    assert audit["all_rows"] == 160
    assert not any("R_best" in f or "Zstd" in f for f in audit["files"])
