import warnings

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp as scipy_solve_ivp

from optimalcontrol.simulate._ivp import solve_ivp
from optimalcontrol.simulate import _fixed_stepsize_integrators
from optimalcontrol.utilities import approx_derivative


rng = np.random.default_rng(123)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('dt', [1e-01, 1e-02])
def test_Euler(n_states, dt):
    t0 = rng.normal()

    x0 = rng.normal(size=(n_states,))

    def f(t, x):
        return np.cos(t) * x

    for tf in (t0 + rng.uniform(0., 1.), t0 + dt / 2.):
        solver = _fixed_stepsize_integrators.Euler(f, t0, x0, tf, dt=dt)
        solver.step()

        # Compare solver output with expected
        h = np.minimum(tf - t0, dt)
        f0 = f(t0, x0)
        xf_expect = x0 + h * f0

        assert solver.t == t0 + h
        np.testing.assert_allclose(solver.y, xf_expect)

        # Dense output should satisfy the constraints
        #   x(t0) = x0
        #   x(t0 + h) = xf
        #   dx/dt(t0) = f(t0, x0)
        ode_sol = solver.dense_output()

        np.testing.assert_allclose(ode_sol(t0), x0)
        np.testing.assert_allclose(ode_sol(solver.t), solver.y,
                                   atol=1e-12, rtol=1e-06)
        dxdt = approx_derivative(ode_sol, t0, method='cs').flatten()
        np.testing.assert_allclose(dxdt, f0, atol=1e-12, rtol=1e-06)
        
        
@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('dt', [1e-01, 1e-02])
def test_Midpoint(n_states, dt):
    t0 = rng.normal()

    x0 = rng.normal(size=(n_states,))

    def f(t, x):
        return np.cos(t) * x

    for tf in (t0 + rng.uniform(0., 1.), t0 + dt / 2.):
        solver = _fixed_stepsize_integrators.Midpoint(f, t0, x0, tf, dt=dt)
        solver.step()

        # Compare solver output with expected
        h = np.minimum(tf - t0, dt)
        t1 = t0 + h / 2.
        f0 = f(t0, x0)
        k1 = x0 + h / 2. * f0
        f1 = f(t1, k1)
        xf_expect = x0 + h * f1

        assert solver.t == t0 + h
        np.testing.assert_allclose(solver.y, xf_expect)

        # Dense output should satisfy the constraints
        #   x(t0) = x0
        #   x(t0 + h) = xf
        #   dx/dt(t0) = f(t0, x0)
        #   dx/dt(t1) = f(t1, k1)
        ode_sol = solver.dense_output()

        np.testing.assert_allclose(ode_sol(t0), x0)
        np.testing.assert_allclose(ode_sol(solver.t), solver.y,
                                   atol=1e-12, rtol=1e-06)
        dxdt = approx_derivative(ode_sol, t0, method='cs').flatten()
        np.testing.assert_allclose(dxdt, f0, atol=1e-12, rtol=1e-06)
        dxdt = approx_derivative(ode_sol, t1, method='cs').flatten()
        np.testing.assert_allclose(dxdt, f1, atol=1e-12, rtol=1e-06)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('dt', [1., 1e-01, 1e-02])
def test_RK4(n_states, dt):
    t0 = rng.normal()

    x0 = rng.normal(size=(n_states,))

    def f(t, x):
        return np.cos(t) * x

    for tf in (t0 + rng.uniform(0., 1.), t0 + dt / 2.):
        solver = _fixed_stepsize_integrators.RK4(f, t0, x0, tf, dt=dt)
        solver.step()

        # Compare solver output with expected
        h = np.minimum(tf - t0, dt)

        k1 = f(t0, x0)
        k2 = f(t0 + h / 2., x0 + h / 2. * k1)
        k3 = f(t0 + h / 2., x0 + h / 2. * k2)
        k4 = f(t0 + h, x0 + h * k3)
        xf_expect = x0 + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        assert solver.t == t0 + h
        np.testing.assert_allclose(solver.y, xf_expect)

        # Dense output should satisfy the constraints
        #   x(t0) = x0
        #   x(t0 + h) = xf
        #   dx/dt(t0) = f(t0, x0)
        #   dx/dt(t0 + h) = f(t0 + h, xf)
        ode_sol = solver.dense_output()

        np.testing.assert_allclose(ode_sol(t0), x0)
        np.testing.assert_allclose(ode_sol(solver.t), solver.y,
                                   atol=1e-12, rtol=1e-06)
        dxdt = approx_derivative(ode_sol, t0, method='cs').flatten()
        np.testing.assert_allclose(dxdt, k1, atol=1e-12, rtol=1e-06)
        dxdt = approx_derivative(ode_sol, t0 + h, method='cs').flatten()
        np.testing.assert_allclose(dxdt, f(t0 + h, xf_expect),
                                   atol=1e-12, rtol=1e-06)


@pytest.mark.parametrize('method', ['Euler', 'Midpoint', 'RK4'])
@pytest.mark.parametrize('dt', [1e-01, 1e-02])
def test_solve_ivp_fixed_stepsize(method, dt):
    t0 = rng.normal()
    t1 = t0 + 10.

    t_expect = np.arange(t0, t1, dt)
    t_expect = np.concatenate((t_expect, [t1]))

    x0 = rng.normal(size=(1,))

    def f(t, x):
        return np.cos(t) * x

    ode_sol = solve_ivp(f, [t0, t1], x0, method=method, dt=dt)

    np.testing.assert_allclose(ode_sol.t, t_expect, atol=1e-12)

    ref_sol = solve_ivp(f, [t0, t1], x0, method='RK45', t_eval=ode_sol.t,
                        atol=1e-12, rtol=1e-06)

    order = _fixed_stepsize_integrators.METHODS[method].C.shape[0]
    tol = np.maximum(1e-05, 10. * dt ** order)
    np.testing.assert_allclose(ode_sol.y, ref_sol.y, atol=tol, rtol=tol)


@pytest.mark.parametrize('method', ['RK45', 'BDF'])
@pytest.mark.parametrize('eps', [1e-03, 1e-02])
def test_solve_ivp_events(method, eps):
    t1 = 2.
    t_eval = np.linspace(0., t1, 2001)

    x0 = rng.normal(size=(1,))

    c = 0.1

    def f(t, x):
        return c * x

    # Solution with slightly perturbed parameters
    ref_sol = make_interp_spline(t_eval, x0 * np.exp((c + eps) * t_eval),
                                 k=1, axis=-1)

    # Want integration to stop when square error is greater than eps ** 2
    def integration_event(t, x):
        return (ref_sol(t).T - x) ** 2 - eps ** 2

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
