import pytest

import numpy as np

from optimalcontrol.utilities import approx_derivative

rng = np.random.default_rng()

@pytest.mark.parametrize('n_points', range(4))
@pytest.mark.parametrize('n_states', range(1,4))
@pytest.mark.parametrize('n_out', range(4))
def test_approx_derivative(n_states, n_out, n_points):
    if n_points == 0:
        x = rng.uniform(low=-1., high=1., size=(n_states,))
    else:
        x = rng.uniform(low=-1., high=1., size=(n_states, n_points))

    w = np.pi * np.arange(1,n_states+1)
    if n_points > 0:
        w = w.reshape(-1, 1)

    if n_out == 0:
        n_out = 1
        flatten = True
    else:
        flatten = False

    def vector_fun(x):
        f = [np.cos(i * (x * w).sum(axis=0)) for i in range(1,n_out+1)]
        f = np.stack(f, axis=0)
        if flatten:
            return f[0]
        else:
            return f

    f0 = vector_fun(x)

    if n_points == 0:
        if flatten:
            assert f0.ndim == 0
        else:
            assert f0.shape == (n_out,)
    else:
        if flatten:
            assert f0.shape == (n_points,)
        else:
            assert f0.shape == (n_out, n_points)

    # Construct analytical derivatives for comparison
    if n_points == 0:
        dfdx_expected = np.empty((n_out, n_states))
    else:
        dfdx_expected = np.empty((n_out, n_states, n_points))
    for i in range(1,n_out+1):
        dfdx_expected[i-1] = -i * np.sin(i * (x * w).sum(axis=0)) * w
    if flatten:
        dfdx_expected = dfdx_expected[0]

    for method in ['2-point', '3-point', 'cs']:
        dfdx_approx = approx_derivative(vector_fun, x, method=method)
        np.testing.assert_allclose(
            dfdx_approx, dfdx_expected, rtol=1e-03, atol=1e-06
        )
