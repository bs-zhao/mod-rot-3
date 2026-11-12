import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import polynomial_warp_dictionary as polynomial
import joint_warp_source as source_fit
from test_approximate_ux_function import shared_fixture


def test_polynomial_GP_kernel_equals_complete_kernel_and_respects_U_rotations():
    rng=np.random.default_rng(841);w=rng.normal(size=(35,7));q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    mean=w.mean(0);scale=w.std(0);scale[:4]=np.sqrt(np.var(w[:,:4],axis=0).mean())
    dictionary=polynomial.Dictionary(mean,scale,w[:8],4,'cpu')
    phi=dictionary(w).numpy();z=(w-mean)/scale
    expected=sum((z@z.T/7)**k for k in range(5))
    np.testing.assert_allclose(phi@phi.T,expected,rtol=1e-12,atol=1e-12)
    rotated=w.copy();rotated[:,:4]=w[:,:4]@q;rotated_mean=mean.copy();rotated_mean[:4]=mean[:4]@q
    other=polynomial.Dictionary(rotated_mean,scale,rotated[:8],4,'cpu')(rotated).numpy()
    np.testing.assert_allclose(other@other.T,phi@phi.T,rtol=1e-12,atol=1e-12)


def test_actual_row_global_warp_pipeline_accepts_polynomial_factory(tmp_path):
    import json
    source=tmp_path/'source';shared_fixture(source);torch.set_num_threads(2)
    with threadpool_limits(2):
        quality=source_fit.analyze(source,tmp_path/'out',steps=2,chunk=71,device='cpu',dictionary_factory=polynomial.Dictionary)
    assert quality and all(np.isfinite(float(r['learned_original_Y_validation_nmse'])) for r in quality)
    protocol=json.loads((tmp_path/'out/protocol.json').read_text())
    assert protocol['status']=='COMPLETE' and protocol['all_subjects']==32 and protocol['centers']==0
    with np.load(tmp_path/'out/dictionary.npz') as saved:assert 'polynomial_powers' in saved.files and 'centers_raw' not in saved.files
