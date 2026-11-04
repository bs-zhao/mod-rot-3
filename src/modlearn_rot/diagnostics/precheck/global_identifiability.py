import numpy as np
from scipy.optimize import minimize


def run_global_identifiability(
    model,
    rng,
    n_targets=12,
    n_starts=8,
    n_design=2048,
    exclusion_radius=0.15,
    collision_tol=1e-8,
):
    """Search explicitly for a distant parameter vector with the same function.

    Earlier code minimized function distance without enforcing distance from the
    target; for identifiable models every optimizer simply returned to the true
    target and the diagnostic reported Infinity.  Here SLSQP enforces the
    exclusion radius, so the reported best distant function distance is finite
    and directly interpretable.
    """
    x=model.sample_inputs(n_design,rng)
    targets=model.sample_parameters(n_targets,rng)
    # Scientific parameterizations such as CPT require constrained domains.
    # Preserve the original unconstrained diagnostic for models without bounds.
    bounds=getattr(model,"parameter_bounds",None)
    out=[]
    for target in targets:
        target=np.asarray(target,dtype=float)
        tf=np.asarray(model.predictive_object(x,target),dtype=float).reshape(-1)

        def dist(p):
            f=np.asarray(model.predictive_object(x,p),dtype=float).reshape(-1)
            return float(np.mean((f-tf)**2))

        def constraint(p):
            return float(np.linalg.norm(np.asarray(p)-target)-exclusion_radius)

        starts=model.sample_parameters(max(n_starts*4,n_starts),rng)
        distant=[
            np.asarray(s,dtype=float)
            for s in starts
            if np.linalg.norm(np.asarray(s)-target)>=exclusion_radius
        ][:n_starts]
        if not distant:
            direction=rng.normal(size=model.n_factors)
            direction/=max(np.linalg.norm(direction),1e-15)
            distant=[target+(exclusion_radius*1.5)*direction]

        best=np.inf
        best_pd=np.inf
        success_count=0
        for start in distant:
            result=minimize(
                dist,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=({"type":"ineq","fun":constraint},),
                options={"maxiter":500,"ftol":1e-12},
            )
            candidate=np.asarray(result.x,dtype=float)
            pd=float(np.linalg.norm(candidate-target))
            fd=dist(candidate)
            if result.success and pd>=exclusion_radius*(1.0-1e-6):
                success_count+=1
                if fd<best:
                    best=fd
                    best_pd=pd

        # Always provide a finite empirical fallback if constrained optimization
        # fails numerically.
        if not np.isfinite(best):
            for candidate in distant:
                pd=float(np.linalg.norm(candidate-target))
                fd=dist(candidate)
                if fd<best:
                    best=fd
                    best_pd=pd

        out.append({
            "best_distant_function_distance":float(best),
            "best_distant_parameter_distance":float(best_pd),
            "constrained_optimizations_succeeded":int(success_count),
            "collision_found":bool(best<collision_tol),
        })

    best_distances=np.asarray(
        [row["best_distant_function_distance"] for row in out],
        dtype=float,
    )
    return {
        "targets_checked":int(n_targets),
        "collision_tolerance":float(collision_tol),
        "exclusion_radius":float(exclusion_radius),
        "minimum_distant_function_distance":float(np.min(best_distances)),
        "median_distant_function_distance":float(np.median(best_distances)),
        "results":out,
        "passed":not any(row["collision_found"] for row in out),
    }
