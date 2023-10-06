import warnings

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp as scipy_solve_ivp

from optimalcontrol.simulate import integrate_fixed_time, integrate_to_converge
from optimalcontrol.simulate._ivp import solve_ivp
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params


@pytest.mark.parametrize('method', ['RK45', 'BDF'])
@pytest.mark.parametrize('eps', [1e-04, 1e-03, 1e-02])
def test_solve_ivp_events(method, eps):
    t1 = 2.
    t_eval = np.linspace(0., t1, 2001)

    x0 = np.random.default_rng(123).normal(size=(1,))

    c = 0.1

    def f(t, x):
        return c * x

    # Solution with slightly perturbed parameters
    ref_sol = interp1d(t_eval, x0 * np.exp((c + eps) * t_eval))

    # Want integration to stop when square error is greater than eps ** 2
    def integration_event(t, x):
        return (ref_sol(t) - x) ** 2 - eps ** 2

    # The event condition starts negative, integration should stop when this
    # becomes positive
    integration_event.direction = 1
    integration_event.terminal = True

    kwargs = {'t_eval': t_eval, 'method': method, 'events': integration_event}
    args = (f, [0., t1], x0)

    with warnings.catch_warnings():
        # Silence warning about 1d array to scalar conversion
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        ref_ode_sol = scipy_solve_ivp(*args, **kwargs)

    ode_sol = solve_ivp(*args, exact_event_times=True, **kwargs)

    np.testing.assert_array_equal(ode_sol.t, ref_ode_sol.t)
    np.testing.assert_array_equal(ode_sol.y, ref_ode_sol.y)

    assert not np.any(integration_event(ode_sol.t, ode_sol.y) > 0.)


@pytest.mark.parametrize('method', ['RK45', 'BDF'])
@pytest.mark.parametrize('u_ub', (None, 0.5))
def test_integrate_closed_loop_lqr(method, u_ub):
    """
    Basic test of an LQR-controlled linear system integrated over a fixed time
    horizon. Since the closed-loop system should be stable, checks that the
    system is close to equilibrium after a reasonably long time horizon.
    """
    n_states = 3
    n_controls = 2
    t_span = [0., 60.]
    t_eval = np.linspace(t_span[0], t_span[-1], 3001)

    u_lb = None if u_ub is None else -u_ub

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=123)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                 u_lb=u_lb, u_ub=u_ub, x0_lb=-1., x0_ub=1.,
                                 x0_sample_seed=456)
    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                   u_lb=u_lb, u_ub=u_ub)

    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    t, x, status = integrate_fixed_time(ocp, lqr, x0, t_span, t_eval=t_eval,
                                        method=method, atol=1e-12, rtol=1e-06)
    u = lqr(x)
    cost = ocp.running_cost(x, u)

    assert status == 0
    np.testing.assert_allclose(t[0], t_span[0])
    np.testing.assert_allclose(t[-1], t_span[-1])

    # At large final time, system should be close to equilibrium
    f_tf = ocp.dynamics(x[:, -1], u[:, -1])
    assert np.linalg.norm(f_tf) < 1e-03
    np.testing.assert_allclose(x[:, -1], xf.flatten(), atol=1e-03, rtol=1e-02)
    np.testing.assert_allclose(u[:, -1], uf.flatten(), atol=1e-03, rtol=1e-02)
    np.testing.assert_array_less(cost[-1], 1e-06)

    # Expect integrated cost to be close to LQR value function
    xPx = (x[:, :1] - xf).T @ lqr.P @ (x[:, :1] - xf)
    J = ocp.total_cost(t, x, u)[-1]
    np.testing.assert_allclose(xPx, J, atol=1e-02, rtol=1e-02)


@pytest.mark.parametrize('norm', [1, 2, np.inf])
@pytest.mark.parametrize('method', ['RK45', 'BDF'])
@pytest.mark.parametrize('u_ub', (None, 0.5))
def test_integrate_to_converge_lqr(norm, method, u_ub):
    """
    Basic test of an LQR-controlled linear system integrated over an infinite
    (i.e. very long) time horizon. Since the closed-loop system should be
    stable, checks that the system reaches equilibrium to specified tolerance.
    """
    n_states = 3
    n_controls = 2

    u_lb = None if u_ub is None else -u_ub

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=123)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                 u_lb=u_lb, u_ub=u_ub, x0_lb=-1., x0_ub=1.,
                                 x0_sample_seed=456)
    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                   u_lb=u_lb, u_ub=u_ub)

    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    # Check that integration over a very short time horizon doesn't converge
    t_int = .5
    t_max = .75
    t, x, status = integrate_to_converge(ocp, lqr, x0, t_int, t_max,
                                         norm=norm, method=method)

    assert t[-1] >= t_max
    assert status == 2

    # Check that integration over a longer time horizon converges
    t_int = 1.
    t_max = 300.
    ftol = 1e-03
    t, x, status = integrate_to_converge(ocp, lqr, x0, t_int, t_max, norm=norm,
                                         ftol=ftol, method=method,
                                         atol=ftol*1e-06, rtol=ftol*1e-03)

    assert t[-1] < t_max
    assert status == 0
    f_tf = ocp.dynamics(x[:, -1], lqr(x[:, -1]))
    assert np.linalg.norm(f_tf, ord=norm) < ftol

    # Check that times and states are in correct order
    idx = np.argsort(t)
    np.testing.assert_allclose(t, t[idx])
    np.testing.assert_allclose(x, x[:, idx])


@pytest.mark.parametrize('method', ['RK45', 'BDF'])
def test_integrate_to_converge_ftol_array(method):
    r"""
    Test that `integrate_to_converge` with a vector `ftol` converges differently
    for each state. Consider a closed loop system

    $dx_1/dt = - k_1 x_1$

    $dx_2/dt = - k_2 x_2$

    where $k_1, k_2 > 0$. The analytical solution is

    $x_1(t) = x_1(0) \exp(-k_1 * t)$

    $x_2(t) = x_2(0) \exp(-k_2 * t)$

    This means that at any time $t$,

    $dx_1/dt (t) = - k_1 x_1(0) * \exp(-k_1 t)$

    $dx_2/dt (t) = - k_2 x_2(0) * \exp(-k_2 t)$

    For tolerances $\epsilon_1$ and $\epsilon_2$, we have
    $|dx_1/dt (t)| < \epsilon_1$ for
    $t > t_1 = - \log(\epsilon_1 / |k_1 * x_1(0)|) / k_1$

    and $|dx_2/dt (t)| < \epsilon_2$ for

    $t > t_2 = - \log(\epsilon_2 / |k_2 * x_2(0)|) / k_2$.

    Now suppose $k_1 = 1, k2 = 0.1, x_1(0) = 100, x_2(0) = 1,$
    $\epsilon_1 = 10^{-08}$, and $\epsilon_w = 10^{-02}$. Then
    $t_1 = t_2 = 10 log(10) = 23.0259$. We test that `integrate_to_converge`
    completes integration of this system at this time.
    """
    ftol = [1e-08, 1e-02]
    x0 = np.array([100., 1.])

    # Make an uncontrolled linear system with one fast and one slow eigenvalue
    A = np.diag([-1., -0.1])
    B = np.zeros((2, 1))
    Q = np.eye(2)
    R = np.eye(1)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.)
    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)

    t_int = 0.1
    t_max = 30.
    t, x, status = integrate_to_converge(ocp, lqr, x0, t_int, t_max, ftol=ftol,
                                         method=method, atol=1e-12, rtol=1e-06)

    assert status == 0
    np.testing.assert_allclose(t[-1], 10. * np.log(10.), atol=t_int)
