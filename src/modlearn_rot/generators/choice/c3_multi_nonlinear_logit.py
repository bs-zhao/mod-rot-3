"""C3: two independent quadratic effects plus a mixed X1*X2 effect.

A paired-options binary-choice benchmark extending C2. The neural process
receives ONLY the four raw option attributes, never a design basis or b.
The logit is nonlinear in observed X but linear in six subject parameters.
"""
from __future__ import annotations

import numpy as np

from .c2_shared_x_quadratic import ChoiceC2SharedXQuadratic


class ChoiceC3MultiNonlinearLogit(ChoiceC2SharedXQuadratic):
    """Six-factor nested extension of C2 with three nonlinear directions.

    b1: global choice bias
    b2: option-A minus option-B linear X1 effect
    b3: option-A minus option-B quadratic X1 effect
    b4: option-A minus option-B linear X2 effect
    b5: option-A minus option-B quadratic X2 effect
    b6: option-A minus option-B mixed X1*X2 effect

    The inherited C1 logistic MLE, score Jacobian and Fisher are valid for
    any n_factors provided design_matrix and linear_predictor are overridden
    consistently. Both are overridden here because C1 checks b.shape[-1]==4.
    """

    model_id = "C3"
    model_version = "v1"
    n_factors = 6
    parameter_names = ("b1", "b2", "b3", "b4", "b5", "b6")

    n_subjects = 5000
    trial_count_min = 300
    trial_count_max = 500
    n_input_features = 4
    parameter_sd = 0.5

    equation = (
        "eta=b1+sqrt(3)*b2*(X1_A-X1_B)"
        "+sqrt(5)/2*b3*((3*X1_A^2-1)-(3*X1_B^2-1))"
        "+sqrt(3)*b4*(X2_A-X2_B)"
        "+sqrt(5)/2*b5*((3*X2_A^2-1)-(3*X2_B^2-1))"
        "+3*b6*(X1_A*X2_A-X1_B*X2_B);"
        "P(Y=1|X,b)=sigmoid(eta); Y~Bernoulli(p)"
    )
    description = (
        "Six-factor binary option-comparison choice, nesting C2: global "
        "bias, two linear slopes, independent quadratic X1 and X2 effects, "
        "and a mixed X1*X2 effect. Tests whether second-order functional "
        "operators identify three directions unanchored by population-mean "
        "level and first-order effects."
    )
    topology_tags = (
        "global", "one-X-multiple-b", "one-b-multiple-X",
        "multiple-nonlinear-factors", "X1-X2-interaction",
    )
    functional_form_tags = (
        "linear-and-quadratic-X", "mixed-second-order-X", "sigmoid",
        "linear-in-b",
    )
    parameter_role_tags = (
        "choice-bias", "linear-slope", "quadratic-curvature",
        "linear-slope", "quadratic-curvature", "attribute-interaction",
    )

    @staticmethod
    def design_matrix(x: np.ndarray) -> np.ndarray:
        """Return six oracle-only design functions; never supplied to NP."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(
                "C3 expects X with shape (trials, 4): "
                "(X1_A, X1_B, X2_A, X2_B)."
            )
        x1a, x1b, x2a, x2b = x.T
        s3 = np.sqrt(3.0)
        qscale = np.sqrt(5.0) / 2.0
        return np.column_stack((
            np.ones(len(x), dtype=float),
            s3 * (x1a - x1b),
            qscale * ((3.0 * x1a**2 - 1.0) -
                      (3.0 * x1b**2 - 1.0)),
            s3 * (x2a - x2b),
            qscale * ((3.0 * x2a**2 - 1.0) -
                      (3.0 * x2b**2 - 1.0)),
            3.0 * (x1a * x2a - x1b * x2b),
        ))

    def linear_predictor(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if b.ndim == 0 or b.shape[-1] != self.n_factors:
            raise ValueError("C3 requires b with six coefficients.")
        basis = self.design_matrix(x)
        if b.ndim == 1:
            return basis @ b
        return np.einsum("nd,...d->...n", basis, b)

    def manifest_extras(self) -> dict:
        return {
            "parameter_prior": "b1..b6 iid Normal(0,0.5^2)",
            "parameter_sd": self.parameter_sd,
            "input_columns": ["X1_A", "X1_B", "X2_A", "X2_B"],
            "input_distribution": "Four independent Uniform(-1,1) raw columns, trialwise",
            "observation_distribution": "Bernoulli(sigmoid(eta)); 1=option A",
            "num_choices": self.num_choices,
            "design_feature_second_moment": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "design_basis_visible_to_np": False,
            "parent_generating_model": "C2 (nested for b5=b6=0)",
            "nonlinearity_scope": "Nonlinear in X; linear in six b parameters",
            "functional_identifiability_rationale": (
                "Under symmetric iid Uniform(-1,1) option attributes, the "
                "population means of level and first-X derivative signatures "
                "directly anchor b1,b2,b4 but not b3,b5,b6. The latter "
                "have distinct diagonal and mixed second-X signatures. "
                "Finite-trial, trained-decoder v1 may still acquire indirect "
                "information; its failure is not assumed."
            ),
            "balance_rationale": (
                "Six raw-design features are orthogonal in the population, "
                "with E[phi_j^2]=(1,2,2,2,2,2). Equal-variance parameter "
                "priors make the five attribute-dependent factors comparably "
                "scaled without adjusting the C2 base coefficients."
            ),
        }


__all__ = ["ChoiceC3MultiNonlinearLogit"]
