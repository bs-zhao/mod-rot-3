"""Complete port inventory of regression and DDM rotation attempts."""
import numpy as np
import torch
from .regression_reference import (linear_xz_signatures, level_signature,
    nonlinear_basis_signatures, contrast_signatures, rich_x_basis_signatures)
from .entropy_reference import canonical_sign, orthogonal_complement, prepare_rows
from .shape_reference import (fit_vector_additive_splines, detect_shared_x_groups,
                             curvature_refinement_rotation)
from .optimization import RowObjective, optimize, scaled_jacobian, varimax, jvolmax, jvol_score
from .fields import fisher_rows


def regression_attempts(x, subject, field, cfg, emit):
    """Run the same historical criteria on p (mean analogue) and eta (link control)."""
    k = field['logit'].shape[1]
    rng = np.random.default_rng(cfg['seed']+1000)
    sampled = rng.choice(len(x),min(len(x),cfg['gradient_rows']),replace=False)
    joint_sample = sampled[:min(len(sampled),cfg['joint_gradient_rows'])]
    for channel in ['probability','logit']:
        g = field[channel]
        level = level_signature(g,subject)
        linear, linear_r2 = linear_xz_signatures(x,g,subject)
        nonlinear, non_r2, non_names, non_alpha = nonlinear_basis_signatures(x,g,subject,cfg['n_knots'])
        contrast, contrast_names = contrast_signatures(x,g,subject,cfg['n_bins'])
        rich, rich_r2, rich_names, rich_alpha, varying = rich_x_basis_signatures(x,g,subject,cfg['n_knots'])
        blocks = {
            'gradient_geomin':[g[sampled]],
            'linear_xz_geomin':[linear],
            'level_linear_xz_geomin':[level,linear],
            'nonlinear_basis_xz_geomin':[level,nonlinear],
            'nonparametric_contrast_geomin':[level,contrast],
            'joint_signature_geomin':[level,linear,nonlinear,contrast,g[joint_sample]],
            'basis_geomin':[level,rich],
            'basis_joint_geomin':[level,rich,g[sampled]],
        }
        diagnostics = dict(channel=channel, linear_gradient_r2=linear_r2,
            nonlinear_gradient_r2=non_r2, nonlinear_basis_names=non_names,
            nonlinear_ridge_alpha=non_alpha, rich_gradient_r2=rich_r2,
            rich_basis_names=rich_names, rich_ridge_alpha=rich_alpha,
            varying_x_columns=varying,contrast_names=contrast_names,
            fitting_observed_x_only=True, geomin_eps=cfg['regression_geomin_eps'])
        for index,(name,b) in enumerate(blocks.items()):
            score = RowObjective(b,eps=cfg['regression_geomin_eps'])
            a, fit, starts = optimize(score,k,cfg,100+index)
            emit(f'{channel}__{name}',a,dict(**fit,signature=diagnostics),starts)

        # A2b: fixed mean/level direction and entropy only in its complement.
        norm = np.linalg.norm(level)
        if norm < 1e-10:
            raise ValueError(f'{channel}: historical level anchor is undefined')
        direction = canonical_sign(level.reshape(-1)/norm)
        complement = orthogonal_complement(direction)
        perp = rich @ complement.T
        kept, keep, weight, energy, cutoff = prepare_rows(perp,cfg['row_min_frac'])
        anchored = {}
        for name,lam in [('anchored_entropy',0.),('anchored_entropy_coverage',cfg['coverage_lambda'])]:
            score = RowObjective([kept],kind='entropy',weights=[weight],
                                 coverage_lambda=lam,coverage_tau=cfg['coverage_tau'])
            r, fit, starts = optimize(score,k-1,cfg,400)
            a = np.vstack([direction,r@complement])
            meta = dict(**fit,signature=diagnostics,level_norm=float(norm),
                        coverage_lambda=lam,coverage_tau=cfg['coverage_tau'],
                        rows_kept=int(keep.sum()),row_cutoff=cutoff,
                        level_fixed=True)
            full_starts = np.stack([np.vstack([direction,s@complement]) for s in starts])
            emit(f'{channel}__{name}',a,meta,full_starts)
            anchored[name] = a

        # A2c: whole-function curvature, only inside automatically shared-X groups.
        a0 = anchored['anchored_entropy_coverage']
        fit = fit_vector_additive_splines(x,g@a0.T,subject,
                                        n_knots=cfg['shape_knots'],grid_n=cfg['shape_grid'])
        groups, axis_diagnostics, _, _ = detect_shared_x_groups(fit,
                min_effect_fraction=cfg['shape_effect_fraction'],min_dominance=cfg['shape_dominance'])
        s, group_info, error = curvature_refinement_rotation(k,groups,fit)
        emit(f'{channel}__functional_shape',s@a0,
             dict(shared_groups=groups,axis_diagnostics=axis_diagnostics,
                  group_refinements=group_info,gradient_fit_r2=fit['fit_r2'],
                  orthogonality_error=error, no_groups_is_exact_noop=not bool(groups)),None)


def ddm_attempts(common, integrated, observed, integration_weight, observed_weight, cfg, emit):
    k = common['logit'].shape[-1]
    # Choice-only distribution fingerprint: scalar choice log-odds, common-X features.
    # There is no RT data or decoder in these four benchmarks.
    rows,rms,keep = scaled_jacobian(common['fd_logit'])
    _,singular,vt = np.linalg.svd(rows,full_matrices=False)
    projection = vt.T
    r,fit = varimax(rows@projection)
    emit('ddm__fingerprint_svd_varimax',(projection@r).T,
         dict(**fit,singular_values=singular,feature_rms=rms,feature_keep=keep,
              summary='choice log-odds only; central finite differences',fd_step=cfg['fd_step']),None)
    a,fit,starts = jvolmax(rows,cfg)
    emit('ddm__jvolmax_oblique',a,fit,starts)
    row_t = torch.tensor(rows)
    score = lambda a: -jvol_score(a.T,row_t,cfg['jvol_p'])
    a,fit,starts = optimize(score,k,cfg,7001)
    emit('ddm__jvolmax_orthogonal',a,dict(**fit,oblique=False),starts)

    fields = [('fixed_common_x',common,np.full(common['p'].shape,1/common['p'].size)),
              ('integrated_x',integrated,np.broadcast_to(integration_weight[None,:]/len(integrated['p']),integrated['p'].shape)),
              ('observed_x',observed,observed_weight)]
    for index,(design,field,weight) in enumerate(fields):
        b,w,info = fisher_rows(field,weight,cfg['fisher_trace_floor'])
        for criterion in ['geomin','quartimin']:
            score = RowObjective([b],kind=criterion,eps=cfg['fisher_geomin_eps'],weights=[w])
            a,fit,starts = optimize(score,k,cfg,900+index)
            name = f'ddm__{design}_fisher_{criterion}'
            emit(name,a,dict(**fit,fisher=info,components=['choice'],
                 normalized_by_trace=True, geomin_eps=cfg['fisher_geomin_eps']),starts)
            if design == 'integrated_x':
                # Exact mathematical degeneracy, not a second nominal experiment.
                emit(f'ddm__integrated_x_joint_fisher_{criterion}',a,
                     dict(alias_of=name,reason='Only one predictive component; joint equals channelwise'),None)
