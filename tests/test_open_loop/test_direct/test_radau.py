import os

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.open_loop.direct import radau, time_maps


rng = np.random.default_rng()

# Reference data computed with Matlab scripts
test_data_path = os.path.join('tests', 'test_open_loop', 'test_direct',
                              'lgr_diff_data')
lgr_diff_reference = dict()

for fn in os.listdir(test_data_path):
    if fn.endswith('.csv'):
        n_nodes = int(fn[1:-4])
        lgr_diff_reference[n_nodes] = dict()

        lgr_diff_data = pd.read_csv(os.path.join(test_data_path, fn))

        for key in ['tau', 'w']:
            lgr_diff_reference[n_nodes][key] = lgr_diff_data[key]

        cols = [c for c in lgr_diff_data.columns if c[0] == 'D']
        lgr_diff_reference[n_nodes]['D'] = lgr_diff_data[cols]


@pytest.mark.parametrize('n', [-1, 0, 1, 2])
@pytest.mark.parametrize('fun', [radau.make_lgr_nodes, radau.make_lgr])
def test_make_lgr_small_n(n, fun):
    with pytest.raises(ValueError):
        fun(n)


@pytest.mark.parametrize('n', [0, 1, 2])
@pytest.mark.parametrize('fun', [radau.make_lgr_weights,
                                 radau.make_lgr_diff_matrix])
def test_make_lgr_small_tau(n, fun):
    tau = rng.uniform(size=n)
    with pytest.raises(ValueError):
        fun(tau)


@pytest.mark.parametrize('n', sorted(lgr_diff_reference.keys()))
def test_make_lgr(n):
    tau, w, D = radau.make_lgr(n)
    np.testing.assert_allclose(tau, lgr_diff_reference[n]['tau'])
    np.testing.assert_allclose(w, lgr_diff_reference[n]['w'])
    np.testing.assert_allclose(D, lgr_diff_reference[n]['D'])


@pytest.mark.parametrize('n', range(3, 17))
def test_lgr_basic_int_diff(n):
    """
    Tests some basic identities: `w @ x == 0`, `D @ x == ones`,
    `D @ (x ** 2) == 2 * x`, and `w @ (D @ x) == 2`.
    """
    tau, w, D = radau.make_lgr(n)

    np.testing.assert_allclose(np.dot(w, tau), 0., atol=1e-10)
    np.testing.assert_allclose(np.matmul(D, tau), np.ones_like(tau))
    np.testing.assert_allclose(np.matmul(D, tau ** 2), 2. * tau)
    np.testing.assert_allclose(np.dot(w, np.matmul(D, tau)), 2.)


@pytest.mark.parametrize('n', np.arange(3, 17))
def test_lgr_integrate(n):
    """
    LGR should be able to integrate a polynomial of degree `2 * n - 2` to
    machine precision.
    """
    # Generate a random polynomial of degree 2n-2
    degree = 2 * n - 2
    coef = rng.normal(size=degree + 1)
    P = np.polynomial.polynomial.Polynomial(coef)
    expected_integral = P.integ(lbnd=-1.)(1.)

    tau = radau.make_lgr_nodes(n)
    w = radau.make_lgr_weights(tau)
    lgr_integral = np.dot(w, P(tau))

    np.testing.assert_allclose(lgr_integral, expected_integral)


@pytest.mark.parametrize('n', range(3, 17))
def test_lgr_differentiate(n):
    """
    LGR should be able to differentiate a polynomial of degree `n - 1` to
    machine precision.
    """
    # Generate a random polynomial of degree n-1
    degree = n - 1
    coef = rng.normal(size=degree + 1)
    poly = np.polynomial.polynomial.Polynomial(coef)

    tau, w, D = radau.make_lgr(n)
    lgr_derivative = np.matmul(D, poly(tau))

    np.testing.assert_allclose(lgr_derivative, poly.deriv()(tau), atol=1e-12)


@pytest.mark.parametrize('n_dims', [1, 2, 3])
@pytest.mark.parametrize('n', [10, 11])
def test_lgr_multivariate_integrate(n, n_dims):
    """
    Test integration of degree `2 * n - 2` polynomials in `n_dims` dimensions.
    """
    # Generate random polynomial of degree 2 * n - 2
    degree = 2 * n - 2
    coef = rng.normal(size=(n_dims, degree + 1))
    P = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_dims)]
    expected_integral = [P[d].integ(lbnd=-1.)(1.) for d in range(n_dims)]

    tau = radau.make_lgr_nodes(n)
    w = radau.make_lgr_weights(tau)
    P_mat = np.vstack([P[d](tau) for d in range(n_dims)])
    lgr_integral = np.matmul(P_mat, w)

    np.testing.assert_allclose(lgr_integral, expected_integral)


@pytest.mark.parametrize('n_dims', [1, 2, 3])
@pytest.mark.parametrize('n', [10, 11])
def test_lgr_multivariate_differentiate(n, n_dims):
    """
    Test differentiation of degree `n - 1` polynomials in `n_dims` dimensions.
    """
    # Generate a random polynomial of degree n-1
    degree = n - 1
    coef = rng.normal(size=(n_dims, degree + 1))
    P = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_dims)]

    tau, w, D = radau.make_lgr(n)
    P_mat = np.vstack([P[d](tau) for d in range(n_dims)])

    expected_derivative = [P[d].deriv()(tau) for d in range(n_dims)]
    lgr_derivative = np.matmul(P_mat, D.T)

    np.testing.assert_allclose(lgr_derivative, expected_derivative)


@pytest.mark.parametrize('time_map', (time_maps.TimeMapRational,
                                      time_maps.TimeMapLog,
                                      time_maps.TimeMapLog2))
def test_time_maps(time_map):
    t_orig = np.linspace(0., 10.)
    tau = time_map.physical_to_radau(t_orig)
    t = time_map.radau_to_physical(tau)
    np.testing.assert_allclose(t, t_orig, atol=1e-14)

    r = time_map.derivative(tau)
    for k in range(tau.shape[0]):
        r_num = approx_derivative(time_map.radau_to_physical, tau[k],
                                  method='cs')
        assert(np.isclose(r[k], r_num))
