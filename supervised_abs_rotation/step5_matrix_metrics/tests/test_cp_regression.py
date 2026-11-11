import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cp_regression as cp
import run_cp_regression as runner
from test_matrix_metrics import make_cases


def test_w_only_canonicalization_quotients_signs_and_permutations():
    rng = np.random.default_rng(521); u = rng.normal(size=(100, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4))); p = np.eye(4)[:, [2, 0, 3, 1]]*[-1, 1, 1, -1]
    classes, membership = cp.canonical_classes(u, [q, q @ p, np.eye(4)])
    assert len(classes) == 2
    assert membership[0] == membership[1]


def test_whitened_features_and_warm_rank_embedding_preserve_predictions():
    rng = np.random.default_rng(522)
    up, _, _, au = cp.whiten(cp.features(rng.normal(size=(2, 300, 3))))
    xp, _, _, ax = cp.whiten(cp.features(rng.normal(size=(300, 2))))
    np.testing.assert_allclose(up[..., 1:].mean(1), 0., atol=1e-10)
    for g in range(2):
        for d in range(3):
            gram = up[g, :, d].T @ up[g, :, d]/300
            np.testing.assert_allclose(np.diag(gram), au[g, d].astype(float), atol=1e-7)
    tensors = [torch.tensor(a, dtype=torch.float32) for a in (xp, up, au, ax)]
    model = cp.Model(2, 2, 2, 3, 2, up.shape[-1], 1)
    bigger = cp.Model(2, 2, 2, 3, 2, up.shape[-1], 4, previous=model.snapshot())
    ids = torch.arange(8); subjects = torch.arange(40)
    a = model.predict(ids, subjects, tensors[0][:40], tensors[1], tensors[2], tensors[3])
    b = bigger.predict(ids, subjects, tensors[0][:40], tensors[1], tensors[2], tensors[3])
    torch.testing.assert_close(a, b)


def test_batched_fit_gradients_are_independent_across_cases_and_channels():
    rng = np.random.default_rng(523)
    up, _, _, au = cp.whiten(cp.features(rng.normal(size=(2, 50, 2))))
    xp, _, _, ax = cp.whiten(cp.features(rng.normal(size=(50, 2))))
    model = cp.Model(2, 2, 1, 2, 2, up.shape[-1], 2)
    ids, subjects = torch.tensor([0]), torch.arange(50)
    pred = model.predict(ids, subjects, torch.tensor(xp, dtype=torch.float32), torch.tensor(up, dtype=torch.float32),
                         torch.tensor(au, dtype=torch.float32), torch.tensor(ax, dtype=torch.float32))
    pred.square().mean().backward()
    for parameter in model.parameters():
        assert torch.count_nonzero(parameter.grad[1:]) == 0


def test_full_cp_budget_pipeline_accounts_for_every_original_row(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    source = tmp_path/'source'; make_cases(source)
    output = tmp_path/'out'
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        rows = runner.analyze(source, output, ranks=(1,), epochs=1, starts=1,
                              batch=80, model_chunk=8, device='cpu')
    assert rows
    assert json.loads((output/'protocol.json').read_text())['status'] == 'COMPLETE'
    audit = json.loads((output/'read_audit.json').read_text())
    assert audit['all_original_rows_in_fit_and_metric'] == 160
    assert audit['forbidden_files_read'] == []
