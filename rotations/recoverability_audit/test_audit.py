import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("run.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RecoverabilityTests(unittest.TestCase):
    def test_known_orthogonal_mixture(self):
        rng = np.random.default_rng(13)
        theta = rng.normal(size=(700, 4))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        q = theta @ rotation.T
        r, value = audit.orthogonal_reference(q, theta, starts=4)
        self.assertGreater(value, .999999)
        self.assertTrue(np.allclose(r @ r.T, np.eye(4), atol=1e-10))

    def test_linear_bound_survives_basis_change(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(500, 4))
        theta = q @ rng.normal(size=(4, 4)) + rng.normal(size=(500, 4))*.4
        change = np.eye(4) + rng.normal(size=(4, 4))*.1
        bounds = []
        for coords in (q, q @ change):
            b, _, _ = audit.linear_directions(coords, theta)
            bounds.append(np.diag(audit.correlations(coords @ b, theta))**2)
        self.assertTrue(np.allclose(*bounds))


if __name__ == "__main__":
    unittest.main()
