import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.problem import OptimalControlProblem, LinearQuadraticProblem
from optimalcontrol.open_loop.direct import setup_nlp, radau, time_maps

from tests._utilities import make_LQ_params


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


@pytest.mark.parametrize('n', [11, 12])
@pytest.mark.parametrize('d', [1, 2])
@pytest.mark.parametrize('time_map', (time_maps.TimeMapRational,
                                      time_maps.TimeMapLog,
                                      time_maps.TimeMapLog2))
def test_interp_initial_guess(n, d, time_map):
    """
    Test that the interpolation code recovers the original points if tau = t,
    and that extrapolation uses the last values in the guess.
    """
    t1 = 10.
    t = np.linspace(0., t1, n)
    tau = time_map.physical_to_radau(t)

    x = np.cos(t) * t
    u = np.sin(-t)

    x = np.atleast_2d(x)
    u = np.atleast_2d(u)
    for k in range(d - 1):
        x = np.vstack((x, x[0] + k))
        u = np.vstack((u, u[0] - k))

    t_interp = time_map.radau_to_physical(tau)
    x_interp, u_interp = setup_nlp.interp_guess(t, x, u, t_interp)

    np.testing.assert_allclose(x_interp, x, atol=1e-12)
    np.testing.assert_allclose(u_interp, u, atol=1e-12)

    t_interp = np.linspace(t1, 2. * t1, n - 1)

    x_interp, u_interp = setup_nlp.interp_guess(t, x, u, t_interp)

    for k in range(t_interp.shape[0]):
        np.testing.assert_allclose(x_interp[:, k], x[:, -1], atol=1e-12)
        np.testing.assert_allclose(u_interp[:, k], u[:, -1], atol=1e-12)


@pytest.mark.parametrize('n_states', [1, 2, 3])
@pytest.mark.parametrize('n_controls', [1, 2, 3])
@pytest.mark.parametrize('n_nodes', [12, 13])
@pytest.mark.parametrize('order', ['F', 'C'])
def test_reshaping_funs(n_states, n_controls, n_nodes, order):
    x = rng.normal(size=(n_states, n_nodes))
    u = rng.normal(size=(n_controls, n_nodes))

    xu = setup_nlp.collect_vars(x, u, order=order)
    assert xu.ndim == 1
    assert xu.shape[0] == (n_states + n_controls) * n_nodes

    _x, _u = setup_nlp.separate_vars(xu, n_states, n_controls, order=order)
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
    tau, w, D = radau.make_lgr(n_nodes)

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
    xu = setup_nlp.collect_vars(x, u, order=order)

    constr = setup_nlp.make_dynamic_constraint(ocp, D, order=order)

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


@pytest.mark.parametrize('n_nodes', [3, 4, 5])
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('u_bound', (None, np.inf, 1.5, [1., np.inf]))
@pytest.mark.parametrize('x_bound', (None, np.inf, 2., [np.inf, 2.]))
def test_bounds_setup(n_nodes, order, u_bound, x_bound):
    """
    Test that Bounds are initialized correctly for different kinds of possible
    control bounds.
    """
    n_t = n_nodes

    u_ub = u_bound
    if u_bound is None:
        u_lb = u_ub
        n_u = n_t - 1
    else:
        u_lb = - np.asarray(u_ub)
        n_u = u_lb.size

    x_ub = x_bound
    if x_bound is None:
        x_lb = x_ub
        n_x = n_t - 1
    else:
        x_lb = - np.asarray(x_ub)
        n_x = x_lb.size

    A, B, Q, R, xf, uf = make_LQ_params(n_x, n_u)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                 x0_lb=-1., x0_ub=1.,
                                 u_lb=u_lb, u_ub=u_ub, x_lb=x_lb, x_ub=x_ub)

    x0 = ocp.sample_initial_conditions()

    constr = setup_nlp.make_bound_constraint(ocp, x0, n_t, order=order)

    assert constr.lb.shape == constr.ub.shape == ((n_x + n_u) * n_t,)

    x_lb_c, u_lb_c = setup_nlp.separate_vars(constr.lb, n_x, n_u, order=order)
    x_ub_c, u_ub_c = setup_nlp.separate_vars(constr.ub, n_x, n_u, order=order)

    # Initial condition constraint
    np.testing.assert_allclose(x_lb_c[:, 0], x0, atol=1e-14)
    np.testing.assert_allclose(x_ub_c[:, 0], x0, atol=1e-14)

    # State bounds
    if x_bound is None:
        np.testing.assert_allclose(x_lb_c[:, 1:], -np.inf)
        np.testing.assert_allclose(x_ub_c[:, 1:], np.inf)
    else:
        for k in range(1, n_t):
            np.testing.assert_allclose(x_lb_c[:, k], x_lb, atol=1e-14)
            np.testing.assert_allclose(x_ub_c[:, k], x_ub, atol=1e-14)

    # Control bounds
    if u_bound is None:
        np.testing.assert_allclose(u_lb_c, -np.inf)
        np.testing.assert_allclose(u_ub_c, np.inf)
    else:
        for k in range(n_t):
            np.testing.assert_allclose(u_lb_c[:, k], u_lb, atol=1e-14)
            np.testing.assert_allclose(u_ub_c[:, k], u_ub, atol=1e-14)
