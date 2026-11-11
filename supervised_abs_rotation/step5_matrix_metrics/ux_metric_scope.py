"""Distinguish sparse support from single-component dominance diagnostics."""


def primary(metric):
    return not ('rank1_UX_' in metric or '_tail1_' in metric
                or '_budget_area_' in metric or 'eps300' in metric)
