"""Empirical orthogonal split contrasts for a full multi-output CART tree."""
import numpy as np


def coefficients(estimator):
    tree=estimator.tree_;nodes=np.flatnonzero(tree.children_left>=0)
    left,right=tree.children_left[nodes],tree.children_right[nodes]
    count=tree.weighted_n_node_samples;means=tree.value[:,:,0]
    scale=np.sqrt(count[left]*count[right]/count[nodes])
    detail=scale[:,None]*(means[left]-means[right])
    root=np.sqrt(count[0])*means[0]
    return nodes,tree.feature[nodes],detail,root


def energies(estimator,u,target,projections,d):
    nodes,axes,detail,root=coefficients(estimator);n=len(u)
    residual=target-np.asarray(estimator.predict(u)).reshape(n,-1)
    node=np.zeros((len(nodes)+1,len(projections)));axis=np.zeros((d+1,len(projections)));audits=[]
    for j,p in projections:
        values=detail@p;node[:-1,j]=np.sum(values*values,axis=1)/n
        unassigned=(np.sum((root@p)**2)+np.sum((residual@p)**2))/n
        node[-1,j]=unassigned
        for i in range(d):axis[i,j]=np.sum(node[:-1,j][axes==i])
        axis[-1,j]=unassigned
        full=float(np.sum((target@p)**2)/n)
        relative=abs(node[:,j].sum()-full)/max(full,1e-30)
        if relative>1e-8:raise ArithmeticError('Orthogonal tree detail energies failed the observed-function Parseval audit')
        audits.append(dict(X_column=j,original_energy=full,relative_Parseval_error=relative,
            unassigned_root_plus_leaf_residual_energy_fraction=unassigned/max(full,1e-30),
            retained_function_training_relative_MSE=float(np.sum((residual@p)**2)/n/max(full,1e-30))))
    return node,axis,audits
