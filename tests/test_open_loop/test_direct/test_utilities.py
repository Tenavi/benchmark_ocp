import pytest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.open_loop.direct import utilities
from optimalcontrol.open_loop.direct import legendre_gauss_radau as lgr


TOL = 1e-10


def _generate_dynamics(n_x, n_u, poly_deg=5, random_seed=None):
    rng = np.random.default_rng(random_seed)
    A = rng.normal(size=(n_x, n_x + n_u))
    # Make random polynomials of X and U with no constant term or linear term
    X_coefs = np.hstack((np.zeros((n_x, 2)), np.random.randn(n_x, poly_deg-2)))
    X_polys = [
        np.polynomial.polynomial.Polynomial(X_coefs[i], domain=[-10., 10.])
        for i in range(n_x)
    ]
    U_coefs = np.hstack((np.zeros((n_u,1)), np.random.randn(n_u, poly_deg-1)))
    U_polys = [
        np.polynomial.polynomial.Polynomial(U_coefs[i], domain=[-10., 10.])
        for i in range(n_u)
    ]

    def dynamics(X, U):
        flat_out = X.ndim < 2
        X = X.reshape(n_x, -1)
        U = U.reshape(n_u, -1)

        X_poly = np.vstack(
            np.atleast_2d([X_polys[i](X[i]) for i in range(n_x)])
        )
        U_poly = np.vstack(
            np.atleast_2d([U_polys[i](U[i]) for i in range(n_u)])
        )

        dXdt = np.matmul(A, np.vstack((X_poly, U_poly)))
        if flat_out:
            dXdt = dXdt.flatten()

        return dXdt

    def jacobians(X, U):
        X = X.reshape(n_x, -1)
        U = U.reshape(n_u, -1)
        n_t = X.shape[1]

        dFdX = np.empty((n_x, n_x, n_t))
        dFdU = np.empty((n_x, n_u, n_t))

        for k in range(n_t):
            F0 = dynamics(X[:,k], U[:,k]).flatten()
            dFdX[...,k] = approx_derivative(
                lambda Xk: dynamics(Xk, U[:,k]), X[:,k], f0=F0, method='cs'
            )
            dFdU[...,k] = approx_derivative(
                lambda Uk: dynamics(X[:,k], Uk), U[:,k], f0=F0, method='cs'
            )
        return dFdX, dFdU

    return dynamics, jacobians


@pytest.mark.parametrize('n', [10,15])
@pytest.mark.parametrize('d', [1,2])
def test_interp_initial_guess(n, d):
    '''
    Test that the interpolation code recovers the original points if tau = t.
    '''
    t = np.linspace(0.,10.,n)
    tau = lgr.time_map(t)

    X = np.cos(t) * t
    U = np.sin(-t)

    X = np.atleast_2d(X)
    U = np.atleast_2d(U)
    for k in range(d-1):
        X = np.vstack((X, X[0] + k))
        U = np.vstack((U, U[0] - k))

    X_interp, U_interp = utilities.interp_guess(t, X, U, tau, lgr.time_map)

    assert np.allclose(X_interp, X)
    assert np.allclose(U_interp, U)


@pytest.mark.parametrize('n_states', [1, 3])
@pytest.mark.parametrize('n_controls', [1, 2])
@pytest.mark.parametrize('n_nodes', [16, 17])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_reshaping_funs(n_states, n_controls, n_nodes, order):
    rng = np.random.default_rng()
    x = rng.normal(size=(n_states, n_nodes))
    u = rng.normal(size=(n_controls, n_nodes))

    xu = utilities.collect_vars(x, u, order=order)
    assert xu.ndim == 1
    assert xu.shape[0] == (n_states + n_controls) * n_nodes

    _x, _u = utilities.separate_vars(xu, n_states, n_controls, order=order)
    np.testing.assert_array_equal(_x, x)
    np.testing.assert_array_equal(_u, u)


@pytest.mark.parametrize('order', ['F', 'C'])
def test_dynamics_setup(order):
    """
    Test that the dynamics constraints are instantiated properly. To this end,
    make a random polynomial which represents the true state. Check that the
    constraint function is zero when evaluated for this state, and not zero when
    evaluated on a significantly perturbed state.
    """
    n_x, n_u, n_t = 3, 2, 13
    rng = np.random.default_rng()

    tau, w, D = lgr.make_LGR(n_t)

    # Generate random polynomials of degree n-1 for the state
    coef = rng.normal(size=(n_x, n_t))
    poly_x = [np.polynomial.polynomial.Polynomial(coef[d]) for d in range(n_x)]
    # control is ignored so can be anything
    xu = utilities.collect_vars(np.vstack([p(tau) for p in poly_x]),
                                rng.normal(size=(n_u, n_t)), order=order)

    # The derivative is the polynomial derivative
    def dxdt(x, u):
        return np.vstack([p.deriv()(tau) for p in poly_x])

    constr = utilities.make_dynamic_constraint(dxdt, D, n_x, n_u, order=order)

    assert constr.lb == constr.ub == 0.

    # Check that evaluating the constraint function for the true state returns 0
    np.testing.assert_allclose(constr.fun(xu), 0., atol=TOL)
    # Check that evaluating the constraint function for perturbed states does
    # not return 0
    with pytest.raises(AssertionError):
        xu = xu + rng.normal(scale=10., size=xu.shape)
        np.testing.assert_allclose(constr.fun(xu), 0., atol=TOL)


@pytest.mark.parametrize('n_nodes', [3, 4, 7, 8])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_dynamics_setup_jacobian(n_nodes, order):
    """
    Use numerical derivatives to verify the sparse dynamics constraint Jacobian.
    """
    n_x, n_u, n_t = 3, 2, n_nodes
    rng = np.random.default_rng()

    tau, w, D = lgr.make_LGR(n_t)

    # Generate random states and controls
    x = rng.normal(size=(n_x, n_t))
    u = rng.normal(size=(n_u, n_t))
    xu = utilities.collect_vars(x, u, order=order)

    # Generate some random dynamics
    dxdt, jacobians = _generate_dynamics(n_x, n_u)

    constr = utilities.make_dynamic_constraint(dxdt, D, n_x, n_u, jac=jacobians,
                                               order=order)

    constr_jac = constr.jac(xu)
    expected_jac = approx_derivative(constr.fun, xu, method='cs')

    assert expected_jac.shape == (n_x * n_t, (n_x + n_u) * n_t)

    np.testing.assert_allclose(constr_jac.toarray(), expected_jac, atol=TOL)


@pytest.mark.parametrize('n_nodes', [3, 4, 7, 8])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_init_cond_setup(n_nodes, order):
    """Check that the initial condition matrix multiplication returns the
    correct points."""
    n_x, n_u, n_t = 3, 2, n_nodes
    rng = np.random.default_rng()

    # Generate random states and controls
    x = rng.normal(size=(n_x, n_t))
    u = rng.normal(size=(n_u, n_t))
    xu = utilities.collect_vars(x, u, order=order)
    x0 = x[:, :1]

    constr = utilities.make_initial_condition_constraint(x0, n_u, n_t,
                                                         order=order)

    np.testing.assert_array_equal(constr.lb, x0.flatten())
    np.testing.assert_array_equal(constr.ub, x0.flatten())
    assert constr.A.shape == (n_x, (n_x + n_u)*n_t)
    # Check that evaluating the multiplying the linear constraint matrix
    # times the full state-control vector returns the initial condtion
    np.testing.assert_allclose(constr.A @ xu, x0.flatten(), rtol=TOL)


@pytest.mark.parametrize('n_nodes', [3, 4, 5])
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('u_lb', (None, -1., [-1.], [-1., -2.],
                                  [-np.inf, -np.inf], [-np.inf, -2.]))
def test_bounds_setup(n_nodes, order, u_lb):
    """
    Test that Bounds are initialized correctly for all different kinds of
    possible control bounds.
    """
    if u_lb is None:
        u_ub = None
        n_u = 1
    elif np.isinf(u_lb).all():
        u_lb = None
        u_ub = None
        n_u = 2
    else:
        u_lb = np.reshape(u_lb, (-1, 1))
        u_ub = - u_lb
        n_u = u_lb.shape[0]

    n_x, n_t = 3, n_nodes
    rng = np.random.default_rng()

    constr = utilities.make_bound_constraint(u_lb, u_ub, n_x, n_t, order=order)

    if u_lb is None and u_ub is None:
        assert constr is None
    else:
        assert constr.lb.shape == constr.ub.shape == ((n_x + n_u) * n_t,)

        # No state constraints
        assert np.isinf(constr.lb[:n_x * n_nodes]).all()
        assert np.isinf(constr.ub[:n_x * n_nodes]).all()

        if u_lb is None:
            assert np.isinf(constr.lb[n_x * n_nodes:]).all()
        else:
            u = np.tile(u_lb, (1, n_nodes))
            xu = utilities.collect_vars(rng.normal(size=(n_x, n_t)), u,
                                        order=order)
            np.testing.assert_allclose(constr.lb[n_x * n_nodes:],
                                       xu[n_x * n_nodes:], atol=TOL)

        if u_ub is None:
            assert np.isinf(constr.ub[n_x * n_nodes:]).all()
        else:
            u = np.tile(u_ub, (1, n_nodes))
            xu = utilities.collect_vars(rng.normal(size=(n_x, n_t)), u,
                                        order=order)
            np.testing.assert_allclose(constr.ub[n_x * n_nodes:],
                                       xu[n_x * n_nodes:], atol=TOL)
