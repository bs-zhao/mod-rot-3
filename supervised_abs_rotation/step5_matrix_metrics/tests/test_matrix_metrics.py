from pathlib import Path
import importlib.util
import json
import hashlib

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location("matrix_wave_test_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


sur = module("surrogates")
st = module("structure")
io = module("matrix_io")
runner = module("run")


@pytest.mark.parametrize("kind", ["poly2", "poly3", "rbf"])
def test_derivatives_against_function_values(kind):
    rng = np.random.default_rng(42)
    w = rng.normal(size=(160, 5))
    scaler = sur.Scaler(w, 3)
    estimator = (sur.RBF(scaler, w[:18], 1.3) if kind == "rbf" else
                 sur.Polynomial(scaler, int(kind[-1])))
    coef = rng.normal(size=(estimator.features(w[:1]).shape[1], 2))
    estimator.set_coef(coef)
    probe = w[70:75]
    gradient, hessian = estimator.derivatives(probe)
    step = 2e-4
    f0 = estimator.features(probe) @ coef
    for i in range(5):
        delta = np.zeros(5); delta[i] = step
        fp = estimator.features(probe + delta) @ coef
        fm = estimator.features(probe - delta) @ coef
        np.testing.assert_allclose(gradient[:, :, i], (fp - fm)/(2*step), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(hessian[:, :, i, i], (fp - 2*f0 + fm)/step**2,
                                   atol=2e-5, rtol=2e-5)
        for j in range(i):
            dj = np.zeros(5); dj[j] = step
            fd = ((estimator.features(probe + delta + dj) -
                   estimator.features(probe + delta - dj) -
                   estimator.features(probe - delta + dj) +
                   estimator.features(probe - delta - dj)) @ coef) / (4*step**2)
            np.testing.assert_allclose(hessian[:, :, i, j], fd, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("kind", ["poly2", "poly3", "rbf"])
def test_fit_rotation_equivariance_and_all_subject_refit(kind):
    rng = np.random.default_rng(7)
    w = rng.normal(size=(320, 5)) * np.array([1., 2., .4, 3., 1.])
    sid = np.repeat(np.arange(40), 8)
    y = (w[:, 0]*w[:, 3] + .2*w[:, 1]**2 + np.sin(w[:, 4]))[:, None]
    train = np.arange(30)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = w.copy(); rotated[:, :3] = w[:, :3] @ q

    def fit(data):
        scaler = sur.Scaler(data, 3, row_mask=sid < 30)
        est = (sur.RBF(scaler, data[:24], 1.5) if kind == "rbf" else
               sur.Polynomial(scaler, int(kind[-1])))
        audit = sur.fit_all_rows(est, data, y, sid, train, chunk=71)
        assert audit["final_fit_subjects"] == 40
        assert audit["final_fit_rows"] == 320
        assert audit["validation_subjects"] == 10
        return est

    a, b = fit(w), fit(rotated)
    np.testing.assert_allclose(a.features(w) @ a.coef, b.features(rotated) @ b.coef,
                               atol=2e-8, rtol=2e-8)
    ga, ha = a.derivatives(w[:10]); gb, hb = b.derivatives(rotated[:10])
    expected_g, expected_h = st.rotate_fields(ga[:, 0], ha[:, 0], q, 3)
    np.testing.assert_allclose(expected_g, gb[:, 0], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(expected_h, hb[:, 0], atol=2e-7, rtol=2e-7)


def test_sparse_tails_separate_small_residuals():
    a = st.concentration(np.array([2., .011, .001])**2, "a")
    b = st.concentration(np.array([2., .001, .001])**2, "a")
    assert a["a_tail1_l2"] > b["a_tail1_l2"]
    assert a["a_entropy"] > b["a_entropy"]


def test_sign_permutation_invariance_and_frobenius_controls():
    rng = np.random.default_rng(15)
    g = rng.normal(size=(41, 7))
    h = rng.normal(size=(41, 7, 7)); h = (h + h.swapaxes(1, 2))/2
    p = np.eye(4)[:, [2, 0, 3, 1]] * [-1, 1, -1, 1]
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    base = st.structural_scores(g, h, 4)
    gp, hp = st.rotate_fields(g, h, p, 4)
    signs = st.structural_scores(gp, hp, 4)
    for key in base:
        assert signs[key] == pytest.approx(base[key], rel=1e-12, abs=1e-12)
    gq, hq = st.rotate_fields(g, h, q, 4)
    rot = st.structural_scores(gq, hq, 4)
    for key in base:
        if key.startswith("control_"):
            assert rot[key] == pytest.approx(base[key], rel=1e-12, abs=1e-12)


def test_flat_and_reversed_curves_cannot_pass():
    cases = [dict(id="baseline", pair_1based=None, angle_deg=0)]
    cases += [dict(id=str(a), pair_1based=[1, 2], angle_deg=a) for a in (15, 30, 45, 60, 75, 90)]
    flat = {c["id"]: 1. for c in cases}
    assert not st.evaluate_curve(flat, cases)["all_planes_pass"]
    good = dict(flat)
    for a in (15, 30, 45, 60, 75):
        good[str(a)] = 1 + np.sin(np.deg2rad(2*a))
    assert st.evaluate_curve(good, cases)["all_planes_pass"]
    assert not st.evaluate_curve({k: -v for k, v in good.items()}, cases)["all_planes_pass"]


def make_cases(folder, d=4):
    rng = np.random.default_rng(5)
    n, trials = 32, 5
    u = rng.normal(size=(n, d)); sid = np.repeat(np.arange(n), trials)
    x = rng.normal(size=(n*trials, 2))
    y = u[sid, 0]*x[:, 0] + u[sid, 1]**2*x[:, 1]
    folder.mkdir()
    np.save(folder / "subject_index.npy", sid)
    np.save(folder / "trial_index.npy", np.tile(np.arange(trials), n))
    np.save(folder / "trial_counts.npy", np.full(n, trials))
    np.savez(folder / "Y.npz", logit_nn=y, p_nn=1/(1 + np.exp(-y)))
    cases = [dict(id="baseline", pair_1based=None, angle_deg=0)]
    for i in range(d):
        for j in range(i + 1, d):
            for angle in (15, 30, 45, 60, 75, 90):
                cases.append(dict(id=f"u{i+1}_u{j+1}_{angle:03d}", pair_1based=[i+1, j+1], angle_deg=angle))
    for c in cases:
        q = np.eye(d)
        if c["pair_1based"]:
            i, j = np.asarray(c["pair_1based"]) - 1
            co = np.cos(np.deg2rad(c["angle_deg"])); si = np.sin(np.deg2rad(c["angle_deg"]))
            q[i, i] = co; q[i, j] = -si; q[j, i] = si; q[j, j] = co
        dest = folder / "cases" / c["id"]; dest.mkdir(parents=True)
        np.save(dest / "W.npy", np.column_stack([u[sid] @ q.T, x]).astype(np.float32))
    (folder / "index.json").write_text(json.dumps(dict(model="C3" if d == 6 else "C1", Z_dim=d,
        X_dim=2, n_subjects=n, n_trials=n*trials, complete=True, cases=cases)), encoding="utf-8")
    # Any attempt to read these forbidden files must fail, not silently help.
    for forbidden in ("R_best.npy", "Zstd.npy"):
        (folder / forbidden).write_bytes(b"FORBIDDEN")
    return cases


@pytest.mark.parametrize("d", [4, 6])
def test_complete_data_only_pipeline_and_dimension_counts(tmp_path, d):
    source = tmp_path / "source"; make_cases(source, d)
    out = tmp_path / "out"
    rows = runner.analyze(source, out, channels=("logit_nn",), per_subject=2,
                          models=("poly2", "rbf1"), centers=10, chunk=51)
    assert len(rows) > 20
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["status"] == "COMPLETE"
    audit = json.loads((out / "read_audit.json").read_text())
    assert audit["evaluation_subjects"] == 32
    assert not any("R_best" in f or "Zstd" in f for f in audit["files"])
    assert len(json.loads((out / "reveal.json").read_text())["mapping"]) == (37 if d == 4 else 91)
    with pytest.raises(FileExistsError):
        runner.analyze(source, out)
    with pytest.raises(ValueError, match="all required"):
        runner.compare({"C1": out}, tmp_path / "compare.csv")


def test_x_alignment_corruption_is_rejected(tmp_path):
    source = tmp_path / "source"; make_cases(source)
    data = io.MatrixCases(source, ("logit_nn",))
    other = data.paths[1]
    w = np.load(other); w[-1, -1] += 1
    np.save(other, w)
    with pytest.raises(ValueError, match="Inconsistent"):
        data.coordinates(other)


def comparison_fixtures(tmp_path, rbf_pass):
    paths = {}
    for label, dim in io.DIMS.items():
        folder = tmp_path / label; folder.mkdir(); paths[label] = folder
        (folder / "blind_scores.csv").write_text("synthetic comparison fixture\n")
        digest = hashlib.sha256((folder / "blind_scores.csv").read_bytes()).hexdigest()
        runner.json_file(folder / "protocol.json", dict(model=label, Z_dim=dim,
            n_subjects=5000, seed=381, status="COMPLETE", scores_sha256=digest))
        rows = []
        for estimator in ("poly2", "rbf1"):
            rows.append(dict(estimator=estimator, channel="logit_nn", metric="UX_column_tail1_l2",
                             all_planes_pass=(estimator == "poly2" or rbf_pass),
                             adequate=True, is_control=False, baseline_rank=1,
                             success_planes=15 if dim == 6 else 6, n_planes=15 if dim == 6 else 6))
        runner.csv_file(folder / "metric_summary.csv", rows)
    return paths


@pytest.mark.parametrize("rbf_pass", [False, True])
def test_four_model_screen_requires_both_estimator_families(tmp_path, rbf_pass):
    paths = comparison_fixtures(tmp_path, rbf_pass)
    rows = runner.compare(paths, tmp_path / "comparison.csv")
    assert any(row["cross_family_pass"] for row in rows) == rbf_pass


def test_comparison_rejects_modified_scores_and_wrong_model(tmp_path):
    paths = comparison_fixtures(tmp_path, True)
    protocol = json.loads((paths["C3"] / "protocol.json").read_text())
    protocol["Z_dim"] = 4
    runner.json_file(paths["C3"] / "protocol.json", protocol)
    with pytest.raises(ValueError, match="own completed"):
        runner.compare(paths, tmp_path / "comparison.csv")
    protocol["Z_dim"] = 6
    runner.json_file(paths["C3"] / "protocol.json", protocol)
    (paths["C3"] / "blind_scores.csv").write_text("changed")
    with pytest.raises(ValueError, match="Frozen scores changed"):
        runner.compare(paths, tmp_path / "comparison.csv")
