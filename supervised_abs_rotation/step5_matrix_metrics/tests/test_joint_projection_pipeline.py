import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_joint_projection
from test_matrix_metrics import make_cases


def test_joint_projection_uses_every_original_row_and_freezes_before_reveal(tmp_path):
    source = tmp_path / "source"
    make_cases(source)
    output = tmp_path / "out"
    rows = run_joint_projection.analyze(source, output, degrees=(2, 3), chunk=51, device="numpy")
    assert rows
    assert max(r["endpoint_error"] for r in rows) < 1e-6
    protocol = json.loads((output / "protocol.json").read_text())
    reveal = json.loads((output / "reveal.json").read_text())
    assert protocol["status"] == "COMPLETE"
    assert protocol["scores_sha256"] == reveal["scores_sha256"]
    audit = json.loads((output / "read_audit.json").read_text())
    assert audit["all_subjects"] == 32
    assert audit["all_trial_rows"] == 160
    assert audit["forbidden_files_read"] == []
