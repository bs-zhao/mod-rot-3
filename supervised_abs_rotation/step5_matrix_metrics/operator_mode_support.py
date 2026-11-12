"""Scale-free support of Ui-Xj conditional operator modes, not just edge norms."""
import numpy as np

import tail_support


def scores(spectra):
    """spectra[Uaxis,Xcolumn,mode] contains squared singular values.

    Normalize across every Ui/mode within each X before counting support.
    Weak edges and weak functional modes therefore enter the same small-tail
    budget; there is no arbitrary edge-versus-complexity mixture weight.
    """
    spectra=np.maximum(np.asarray(spectra,dtype=float),0.)
    if spectra.ndim!=3 or not np.isfinite(spectra).all():raise ValueError('Expected finite Ui-X-mode spectra')
    columns=[tail_support.scores(spectra[:,j,:]) for j in range(spectra.shape[1])]
    result={}
    for metric in columns[0]:
        result['UX_operator_columns_mean_'+metric]=float(np.mean([c[metric] for c in columns]))
        result['UX_operator_columns_worst_'+metric]=float(np.max([c[metric] for c in columns]))
    equal=[]
    for j in range(spectra.shape[1]):
        values=spectra[:,j,:];peak=float(values.max(initial=0.))
        normalized=values/peak if peak else values.copy()
        # Canonical reduction order preserves exact axis-permutation equality
        # even for fractional powers of extremely small singular energies.
        total=float(np.sort(normalized,axis=None).sum())
        equal.append(normalized/total if total else normalized)
    result.update({'equal_X_operator_entries_'+key:value for key,value in tail_support.scores(np.stack(equal)).items()})
    return result
