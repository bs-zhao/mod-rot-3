"""Method-specific smoke test: identity."""
import numpy as np
from rotations.${name}.api import RotationInput
import rotations.identity as method

def test_identity_smoke():
    rng = np.random.default_rng(12)
    q = rng.normal(size=(16,4))
    x = rng.normal(size=(16,5,3))
    y = rng.integers(0,2,size=(16,5))
    data = RotationInput(q,x,y,np.ones_like(y,dtype=bool))
    result = method.fit(data,{})
    assert np.array_equal(result.R,np.eye(4))


def test_standalone_suite_saves_R_U_before_theta(tmp_path):
    """Full suite workflow with synthetic FIT ZXY and post-fit true theta."""
    from types import SimpleNamespace
    from sklearn.decomposition import PCA
    from rotations.identity import experiment
    from rotations.identity.api import RotationInput
    rng=np.random.default_rng(20261104)
    z=rng.normal(size=(26,4))
    tr=np.arange(15)
    va=np.arange(15,20)
    te=np.arange(20,26)
    chosen=tr[:9]
    pca=PCA(n_components=4).fit(z[tr])
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,1e-5))[:,None]
    q=(z-pca.mean_)@W.T
    x=rng.normal(size=(len(chosen),3,2))
    y=rng.integers(0,2,size=(len(chosen),3))
    data=RotationInput(q[chosen],x,y,np.ones(y.shape,dtype=bool))

    class TruthOnlyAfterFit:
        @property
        def parameters(self):
            base=tmp_path/"rotations"/"identity"/"synthetic"/"repeat01"
            outputs=list(base.glob("run_*/rotation__U.npz"))
            assert len(outputs)==1, "True theta accessed before U was saved"
            assert (outputs[0].parent/"rotation__matrix.npz").is_file()
            assert (outputs[0].parent/"rotation__fit.json").is_file()
            return z+0.01*rng.normal(size=z.shape)

    source=dict(root=tmp_path,z=z,data=data,W=W,pca=pca,
                chosen=chosen,tr=tr,val=va,te=te,
                dim=4,native_dim=4,dataset=TruthOnlyAfterFit(),
                dataset_sha="s1-test",latent_sha="s5-test",checkpoint_sha="s3-test")
    args=SimpleNamespace(model="CR3",condition="synthetic",repeat=1,
                         representation="full",rotation_dim=4,
                         fit_max_subjects=9,whiten_floor=1e-5,seed=31,
                         no_evaluate=False,dry_run=False,overwrite=False)
    folder=experiment.execute(args,{},method,source)
    assert (folder/"rotation__test_evaluation.json").is_file()
    assert (folder/"rotation__test_corr2.png").is_file()
    with np.load(folder/"rotation__matrix.npz",allow_pickle=False) as out:
        R,A,b=out["R"],out["A"],out["offset"]
    with np.load(folder/"rotation__U.npz",allow_pickle=False) as out:
        U=out["U"]
    assert np.allclose(R,np.eye(4))
    assert np.allclose(U,z@A.T+b)
    assert np.allclose(U[chosen],data.rotated(R))
    assert experiment.execute(args,{},method,source)==folder
