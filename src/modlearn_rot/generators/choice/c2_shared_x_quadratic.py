"""C2: binary choice with linear and quadratic effects sharing X1.

The generative equation follows the model specification:
  eta = b1 + sqrt(3)*b2*(X1_A-X1_B)
        + sqrt(5)/2*b3*[(3*X1_A**2-1)-(3*X1_B**2-1)]
        + sqrt(3)*b4*(X2_A-X2_B).
Each observed row stores (X1_A, X1_B, X2_A, X2_B).
The response label 1 denotes option A. Logistic MLE/Fisher are inherited
from C1's general fixed-design Bernoulli-GLM implementation; importantly,
the learned Neural Process sees the four raw inputs, NOT the design matrix.
"""
from __future__ import annotations

import numpy as np

from .c1_linear_logit import ChoiceC1LinearLogit


class ChoiceC2SharedXQuadratic(ChoiceC1LinearLogit):
    model_id = "C2"
    model_version = "v1"
    n_factors = 4
    parameter_names = ("b1", "b2", "b3", "b4")
    n_subjects = 5000
    trial_count_min = 300
    trial_count_max = 500
    n_input_features = 4
    parameter_sd = 0.5

    equation = (
        "eta=b1+sqrt(3)*b2*(X1_A-X1_B)"
        "+sqrt(5)/2*b3*((3*X1_A^2-1)-(3*X1_B^2-1))"
        "+sqrt(3)*b4*(X2_A-X2_B);"
        "P(Y=1|X,b)=sigmoid(eta), Y~Bernoulli(p)"
    )
    description = (
        "Binary option-comparison choice. X1 has distinct subject-specific "
        "linear and quadratic effects; X2 has a linear effect; b1 is an "
        "X-independent bias. A nonlinear-in-X, linear-in-parameter control."
    )
    topology_tags = ("global", "one-X-multiple-b", "one-b-one-X")
    functional_form_tags = ("linear-and-quadratic-X", "sigmoid")
    parameter_role_tags = ("choice-bias", "linear-slope", "quadratic-curvature")

    @staticmethod
    def design_matrix(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(
                "C2 expects four raw option attributes per trial: "
                "(X1_A, X1_B, X2_A, X2_B)."
            )
        x1_a, x1_b, x2_a, x2_b = x.T
        return np.column_stack((
            np.ones(x.shape[0], dtype=float),
            np.sqrt(3.0) * (x1_a - x1_b),
            (np.sqrt(5.0) / 2.0) * (
                (3.0 * x1_a**2 - 1.0) - (3.0 * x1_b**2 - 1.0)
            ),
            np.sqrt(3.0) * (x2_a - x2_b),
        ))

    def manifest_extras(self):
        return {
            "parameter_prior": "b1..b4 iid Normal(0,0.5^2)",
            "input_columns": ["X1_A", "X1_B", "X2_A", "X2_B"],
            "input_distribution": (
                "Each raw X independently Uniform(-1,1), sampled anew "
                "for each trial and subject."
            ),
            "observation_distribution": "Bernoulli(sigmoid(eta)); 1=option A",
            "parameter_sd": self.parameter_sd,
            "num_choices": self.num_choices,
            "design_feature_second_moment": [1.0, 2.0, 2.0, 2.0],
            "design_basis_visible_to_np": False,
            "balance_rationale": (
                "Under independent Uniform(-1,1) raw option attributes, "
                "the three difference-based features are orthogonal, "
                "zero-mean and have variance 2; the intercept has second "
                "moment 1. This preserves the specified sqrt(3) and "
                "sqrt(5)/2 coefficients, rather than rescaling the model."
            ),
            "nonlinearity_scope": (
                "Nonlinear in observed X1, still linear in individual "
                "parameters b. This is distinct from later C4, where "
                "parameters change the shape of the nonlinear function."
            ),
        }


__all__ = ["ChoiceC2SharedXQuadratic"]
