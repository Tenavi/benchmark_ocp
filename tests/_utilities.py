import numpy as np

from optimalcontrol.utilities import approx_derivative

rng = np.random.default_rng()

def compare_finite_difference(x, jac, fun, method="3-point"):
    expected_jac = approx_derivative(fun, x, method=method)
    np.testing.assert_allclose(jac, expected_jac)

def make_LQ_params(n_states, n_controls):
    """Generate random dynamics matrices `A` and `B` of specified size and
    corresponding positive definite cost matrices `Q` and `R`."""
    A = rng.normal(size=(n_states, n_states))
    B = rng.normal(size=(n_states, n_controls))
    Q = rng.normal(size=(n_states, n_states))
    Q = Q.T @ Q
    R = rng.normal(size=(n_controls, n_controls))
    R = R.T @ R + 1e-12 * np.eye(n_controls)

    xf = rng.uniform(size=(n_states, 1)) - 0.5
    uf = rng.uniform(size=(n_controls, 1)) - 0.5

    return A, B, Q, R, xf, uf