import pytest

import numpy as np

from optimalcontrol.utilities import approx_derivative

rng = np.random.default_rng()

@pytest.mark.parametrize('n_points', range(4))
@pytest.mark.parametrize('n_states', range(1,4))
@pytest.mark.parametrize('n_out', range(1,4))
def test_approx_derivative(n_states, n_out, n_points):
    if n_points == 0:
        x = rng.uniform(low=-1., high=1., size=(n_states,))
    else:
        x = rng.uniform(low=-1., high=1., size=(n_states, n_points))

    w = np.pi * np.arange(1,n_states+1)
    if n_points > 0:
        w = w.reshape(-1, 1)

    def vector_fun(x):
        f = [np.cos(i * (x * w).sum(axis=0)) for i in range(1,n_out+1)]
        return np.stack(f, axis=0)

    f0 = vector_fun(x)

    if n_points == 0:
        assert f0.shape == (n_out,)
    else:
        assert f0.shape == (n_out, n_points)

    # Construct analytical derivatives for comparison
    if n_points == 0:
        dfdx_expected = np.empty((n_out, n_states))
    else:
        dfdx_expected = np.empty((n_out, n_states, n_points))
    for i in range(1,n_out+1):
        dfdx_expected[i-1] = -i * np.sin(i * (x * w).sum(axis=0)) * w

    for method in ['2-point', '3-point', 'cs']:
        dfdx_approx = approx_derivative(vector_fun, x, method=method)
        assert dfdx_approx.shape == dfdx_expected.shape
        assert np.allclose(dfdx_approx, dfdx_expected, rtol=1e-04, atol=1e-06)
