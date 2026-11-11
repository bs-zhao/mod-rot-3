"""Monotone response calibration plus additive numerical dictionaries.

Inspired by alternating conditional expectations, with a fixed identity
mixture to prevent a flat response transformation. No link formula is used.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression

import additive_profiles as ap


def orders(y, train):
    out = []
    for j in range(y.shape[1]):
        ids = train[np.argsort(y[train, j], kind="stable")]
        unique, starts, counts = np.unique(y[ids, j], return_index=True, return_counts=True)
        out.append((ids, unique, starts, counts))
    return out


def calibrate(y, predicted, ordered, identity_mix=.1):
    transformed = np.empty_like(y)
    links = []
    for j, (ids, unique, starts, counts) in enumerate(ordered):
        means = np.add.reduceat(predicted[ids, j], starts)/counts
        monotone = isotonic_regression(means, weights=counts).x
        center = np.average(monotone, weights=counts)
        scale = max(np.sqrt(np.average((monotone-center)**2, weights=counts)), 1e-12)
        original_center = np.average(unique, weights=counts)
        original_scale = max(np.sqrt(np.average((unique-original_center)**2, weights=counts)), 1e-12)
        values = ((1-identity_mix)*(monotone-center)/scale +
                  identity_mix*(unique-original_center)/original_scale)
        center = np.average(values, weights=counts)
        scale = max(np.sqrt(np.average((values-center)**2, weights=counts)), 1e-12)
        values = (values-center)/scale
        transformed[:, j] = np.interp(y[:, j], unique, values)
        links.append((unique, values))
    return transformed, links


def inverse_link(predicted, links):
    out = np.empty_like(predicted)
    for j, (original, transformed) in enumerate(links):
        out[:, j] = np.interp(predicted[:, j], transformed, original)
    return out


def chain(design, y, train, valid, test, *, identity_mix=.1, iterations=12, ridge=1e-3):
    ordered = orders(y, train)
    mean = y[train].mean(0)
    std = np.maximum(y[train].std(0), 1e-12)
    transformed = (y-mean)/std
    links = [(original, (original-mean[j])/std[j]) for j, (_, original, _, _) in enumerate(ordered)]
    inverse_gram = ap.estimate(design[train].T @ design[train], design[train].T, ridge)
    history = []
    best = None
    for iteration in range(iterations+1):
        coef = inverse_gram @ transformed[train]
        predicted = design @ coef
        restored = inverse_link(predicted, links)
        validation_by_output = np.mean(((restored[valid]-y[valid])/std)**2, axis=0)
        test_by_output = np.mean(((restored[test]-y[test])/std)**2, axis=0)
        validation = float(validation_by_output.mean())
        history.append(dict(iteration=iteration, validation_nmse=validation,
                            test_nmse=float(test_by_output.mean())))
        if best is None or validation < best[0]:
            best = (validation, iteration, test_by_output.copy(), coef.copy())
        if iteration < iterations:
            transformed, links = calibrate(y, predicted, ordered, identity_mix)
    return dict(best_iteration=best[1], validation_nmse=best[0],
                test_nmse_by_output=best[2], coef=best[3], history=history)


def fit_case(u, y, split, family, identity_mix=.1, iterations=12):
    train, valid, test = split
    bases = ap.axis_bases(u, train, family)
    design = np.column_stack([np.ones(len(u)), *bases])
    fit = chain(design, y, train, valid, test, identity_mix=identity_mix, iterations=iterations)
    # Refit chosen iteration count on ALL subjects for structural scores.
    all_rows = np.arange(len(u))
    full_bases = ap.axis_bases(u, all_rows, family)
    full_design = np.column_stack([np.ones(len(u)), *full_bases])
    # Use precisely the iteration count chosen on validation subjects.
    ordered = orders(y, all_rows)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-12)
    transformed = (y-mean)/std
    inv = ap.estimate(full_design.T @ full_design, full_design.T, 1e-3)
    for k in range(fit["best_iteration"]+1):
        coef = inv @ transformed
        if k < fit["best_iteration"]:
            transformed, _ = calibrate(y, full_design @ coef, ordered, identity_mix)
    group_energy = []
    component_variation = []
    total_variation = 0.
    at = 1
    for i, group in enumerate(full_bases):
        values = group @ coef[at:at+group.shape[1]]
        group_energy.append(np.mean(values**2, axis=0))
        delta = np.diff(values[np.argsort(u[:, i])], axis=0)
        # Total signed variation versus total absolute variation per output.
        variation = np.abs(delta).sum(0)
        component_variation.append(np.maximum(variation-np.abs(delta.sum(0)), 0.))
        total_variation += float(variation.sum())
        at += group.shape[1]
    energy = np.stack(group_energy, axis=1)
    ordered_energy = np.sort(energy, axis=1)[:, ::-1]
    scores = dict(unknown_link_test_nmse=float(fit["test_nmse_by_output"].mean()),
        identity_link_test_nmse=fit["history"][0]["test_nmse"],
        component_direction_variation=float(np.sum(component_variation)/max(total_variation, 1e-24)))
    for k in (1, 2):
        scores[f"link_component_tail{k}_l2"] = float(np.mean(np.sqrt(
            ordered_energy[:, k:].sum(1)/np.maximum(energy.sum(1), 1e-24))))
    fit.pop("coef")
    fit["test_nmse_by_output"] = fit["test_nmse_by_output"].tolist()
    return scores, fit
