import numpy as np

from optimalcontrol.utilities import approx_derivative


def compare_finite_difference(x, jac, fun, method='3-point',
                              rtol=1e-06, atol=1e-12):
    expected_jac = approx_derivative(fun, x, method=method)
    np.testing.assert_allclose(jac, expected_jac, rtol=rtol, atol=atol)


def make_LQ_params(n_states, n_controls, seed=None):
    """Generate random dynamics matrices `A` and `B` of specified size and
    corresponding positive definite cost matrices `Q` and `R`."""
    rng = np.random.default_rng(seed)

    A = rng.normal(scale=1/2, size=(n_states, n_states))
    B = rng.normal(scale=1/2, size=(n_states, n_controls))
    Q = rng.normal(scale=1/2, size=(n_states, n_states))
    Q = Q.T @ Q
    R = rng.normal(scale=1/2, size=(n_controls, n_controls))
    R = R.T @ R + 1e-10 * np.eye(n_controls)

    xf = rng.uniform(size=(n_states, 1)) - 0.5
    uf = rng.uniform(size=(n_controls, 1)) - 0.5

    return A, B, Q, R, xf, uf
