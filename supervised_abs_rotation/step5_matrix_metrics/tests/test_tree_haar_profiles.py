import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tree_haar_profiles as haar
import conditional_ux as conditional


def test_split_haar_reconstructs_leaf_predictions_and_retains_nonzero_leaf_residual():
    rng=np.random.default_rng(901);u=rng.normal(size=(100,3));target=rng.normal(size=(100,4))
    estimator=DecisionTreeRegressor(max_leaf_nodes=13,random_state=1).fit(u,target)
    nodes,axes,detail,root=haar.coefficients(estimator);tree=estimator.tree_;counts=tree.weighted_n_node_samples
    path=estimator.decision_path(u).toarray();basis=[]
    for node in nodes:
        a,b=tree.children_left[node],tree.children_right[node]
        basis.append(path[:,a]*np.sqrt(counts[b]/counts[a]/counts[node])-
                     path[:,b]*np.sqrt(counts[a]/counts[b]/counts[node]))
    basis=np.array(basis).T
    np.testing.assert_allclose(basis.T@basis,np.eye(len(nodes)),atol=1e-14)
    prediction=np.ones((len(u),1))@(root[None]/np.sqrt(len(u)))+basis@detail
    np.testing.assert_allclose(prediction,estimator.predict(u),atol=1e-14)
    node,axis,audit=haar.energies(estimator,u,target,[(0,np.eye(4))],3)
    np.testing.assert_allclose(node.sum(0),np.mean(np.sum(target*target,axis=1)),atol=1e-12)
    np.testing.assert_allclose(axis.sum(0),node.sum(0),atol=1e-12)
    assert audit[0]['retained_function_training_relative_MSE']>.1


def test_haar_support_is_invariant_to_response_scale_and_orthogonal_output_basis():
    rng=np.random.default_rng(902);u=rng.normal(size=(120,3));target=rng.normal(size=(120,4))
    p=np.linalg.qr(rng.normal(size=(4,4)))[0]
    first=DecisionTreeRegressor(random_state=1).fit(u,target)
    second=DecisionTreeRegressor(random_state=1).fit(u,3.7*target@p)
    one,_,_=haar.energies(first,u,target,[(0,np.eye(4))],3)
    two,_,_=haar.energies(second,u,3.7*target@p,[(0,np.eye(4))],3)
    a,b=conditional.sparsity(one),conditional.sparsity(two)
    np.testing.assert_allclose(a['UX_columns_mean_soft_count_eps003'],b['UX_columns_mean_soft_count_eps003'],rtol=1e-10)
