"""Conditional random-orientation percentile, counting numerical ties conservatively."""
import numpy as np
from scipy.stats import beta


def summarize(losses,reference,family_size=4,alpha=.05,relative_numerical_tolerance=1e-7):
    values=np.asarray(losses,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Supply finite independent random-frame scores')
    if family_size<1 or not 0<alpha<1:raise ValueError('Invalid confidence family')
    tolerance=relative_numerical_tolerance*max(1.,abs(reference))
    below=int(np.sum(values<reference-tolerance));tied=int(np.sum(np.abs(values-reference)<=tolerance))
    favorable=int(np.sum(values<=reference+tolerance));n=len(values)
    # Invert the binomial lower-tail test. Bonferroni needs no independence
    # between the four datasets; the random draws within each are iid.
    upper=1. if favorable==n else float(beta.ppf(1-alpha/family_size,favorable+1,n-favorable))
    return dict(random_frames=n,reference_loss=float(reference),numerical_tolerance=tolerance,
        random_strictly_below_reference=below,random_numerical_ties=tied,
        random_at_least_as_good_as_reference=favorable,
        empirical_random_favorable_tail_percent=100*favorable/n,
        simultaneous_confidence_level=1-alpha,family_size=family_size,
        random_favorable_tail_upper_percent=100*upper,
        within_15_percent_random_tail_bound=upper<=.15,
        minimum_random_loss=float(values.min()),maximum_random_loss=float(values.max()),
        median_random_loss=float(np.median(values)),
        minimum_random_relative_change_percent=100*(float(values.min())-reference)/max(abs(reference),1e-30),
        scope='Haar rotation measure conditional on this fixed dataset/scorer; not subject-population inference or uniqueness',
        ties_count_against_a_high_reference_rank=True)
