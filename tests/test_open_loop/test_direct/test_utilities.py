import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.open_loop.direct import utilities
from optimalcontrol.open_loop.direct import radau as lgr


rng = np.random.default_rng(123)


class PolynomialDynamics(OptimalControlProblem):
    """
    Define a dummy `OptimalControlProblem` with states that evolve polynomially
    in mapped time `tau`. To enable this time-dependence, the first state,
    `x[0]`, is assumed to be equal to `tau`.

    Parameters
    ----------
    coef : (n_states, deg + 1) array
        Polynomial coefficients of `x` in terms of mapped time `tau`, starting
        from constant terms up to `tau ** deg`.
    n_controls : int
        The number of control inputs to accept. Note that these do not affect
        the dynamics at all.
    x0_sample_seed : int, optional
        Random seed to use for sampling initial conditions.
    """
    _required_parameters = {'coef': None, 'n_controls': None}
    _optional_parameters = {'x0_sample_seed': None}

    @property
    def n_states(self):
        return len(self.parameters._poly_x) + 1

    @property
    def n_controls(self):
        return self.parameters.n_controls

    @property
    def final_time(self):
        return np.inf

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'coef' in new_params:
            obj.coef = np.atleast_2d(obj.coef)
            obj._poly_x = list()
            for c in obj.coef:
                obj._poly_x.append(np.polynomial.polynomial.Polynomial(c))

        if not hasattr(obj, '_rng') or 'x0_sample_seed' in new_params:
            obj._rng = np.random.default_rng(
                getattr(obj, 'x0_sample_seed', None))

    def sample_initial_conditions(self, n_samples=1):
        x0 = self.parameters._rng.normal(size=(self.n_states, n_samples))
        # First dimension is mapped time, which starts at tau = -1
        x0[0] = -1.
        if n_samples == 1:
            return x0.flatten()
        return x0

    def running_cost(self, x, u):
        if np.ndim(x) < 2:
            return np.zeros(1)

        return np.zeros(np.shape(x)[1])

    def dynamics(self, x, u):
        tau = x[0]

        dxdt = [np.ones_like(tau)] + [p(tau) for p in self.parameters._poly_x]

        dxdt = np.stack(dxdt, axis=0)

        if np.ndim(x) < 2:
            return dxdt.flatten()

        return dxdt


@pytest.mark.parametrize('n', [10, 15])
@pytest.mark.parametrize('d', [1, 2])
def test_interp_initial_guess(n, d):
    """
    Test that the interpolation code recovers the original points if tau = t.
    """
    t = np.linspace(0., 10., n)
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
    x = rng.normal(size=(n_states, n_nodes))
    u = rng.normal(size=(n_controls, n_nodes))

    xu = utilities.collect_vars(x, u, order=order)
    assert xu.ndim == 1
    assert xu.shape[0] == (n_states + n_controls) * n_nodes

    _x, _u = utilities.separate_vars(xu, n_states, n_controls, order=order)
    np.testing.assert_array_equal(_x, x)
    np.testing.assert_array_equal(_u, u)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('n_controls', [1, 2])
@pytest.mark.parametrize('n_nodes', [9, 10])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_dynamics_setup(n_states, n_controls, n_nodes, order):
    """
    Test that the dynamics constraints are instantiated properly. To this end,
    generate a system whose state evolves as a degree `n_nodes - 1` polynomial
    of mapped time tau. We expect that LGR can differentiate this state to
    machine precision, so the constraint function should be zero when evaluated
    for this state, and not zero when evaluated on a significantly perturbed
    state. We also test that the constraint Jacobian is approximated well using
    finite differences.
    """
    tau, w, D = lgr.make_lgr(n_nodes)

    # If there are n_nodes - 1 coefficients, the polynomial is degree
    # n_nodes - 2, so the state is degree n_nodes - 1
    coef = rng.normal(size=(n_states, n_nodes - 1))
    ocp = PolynomialDynamics(coef=coef, n_controls=n_controls)

    # From now on, the system looks like it has dimension n_states + 1
    n_states = n_states + 1

    x0 = ocp.sample_initial_conditions()

    # Integrate the polynomials analytically to get the state
    x = [tau]
    for d, poly in enumerate(ocp.parameters._poly_x):
        x.append(poly.integ(k=x0[d + 1])(tau))
    x = np.stack(x, axis=0)
    assert x.shape == (n_states, n_nodes)

    # Control is ignored by the dynamics so can be anything
    u = rng.normal(size=(n_controls, n_nodes))
    xu = utilities.collect_vars(x, u, order=order)

    constr = utilities.make_dynamic_constraint(ocp, D, order=order)

    assert constr.lb == constr.ub == 0.

    constr_fun = constr.fun(xu)
    assert constr_fun.shape == (n_states * n_nodes,)

    # Check that evaluating the constraint function for the true state returns 0
    np.testing.assert_allclose(constr_fun, 0., atol=1e-10)
    # Check that evaluating the constraint function for perturbed states does
    # not return 0
    with pytest.raises(AssertionError):
        xu = xu + rng.normal(size=xu.shape)
        np.testing.assert_allclose(constr.fun(xu), 0., atol=1e-10)

    constr_jac = constr.jac(xu)
    expected_jac = approx_derivative(constr.fun, xu, method='cs')

    # Make sure that our baseline has the right shape at least!
    assert expected_jac.shape == (n_states * n_nodes,
                                  (n_states + n_controls) * n_nodes)

    np.testing.assert_allclose(constr_jac.toarray(), expected_jac, atol=1e-05)


@pytest.mark.parametrize('n_nodes', [3, 4, 7, 8])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_init_cond_setup(n_nodes, order):
    """Check that the initial condition matrix multiplication returns the
    correct points."""
    n_x, n_u, n_t = 3, 2, n_nodes

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
    np.testing.assert_allclose(constr.A @ xu, x0.flatten())


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
                                       xu[n_x * n_nodes:], atol=1e-10)

        if u_ub is None:
            assert np.isinf(constr.ub[n_x * n_nodes:]).all()
        else:
            u = np.tile(u_ub, (1, n_nodes))
            xu = utilities.collect_vars(rng.normal(size=(n_x, n_t)), u,
                                        order=order)
            np.testing.assert_allclose(constr.ub[n_x * n_nodes:],
                                       xu[n_x * n_nodes:], atol=1e-10)
