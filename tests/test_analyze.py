import numpy as np
import pytest

from optimalcontrol import analyze
from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.controls import ConstantControl

from examples.van_der_pol import VanDerPol


class SinusoidSystem(OptimalControlProblem):
    _required_parameters = {'freq': None}
    _optional_parameters = {'x0_sample_seed': None}

    @property
    def n_states(self):
        return self.parameters.freq.shape[0]

    @property
    def n_controls(self):
        return self.n_states

    @property
    def final_time(self):
        return np.inf

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'freq' in new_params:
            obj.freq = np.atleast_2d(obj.freq)
            obj.freq = obj.freq.reshape(obj.freq.shape[0], obj.freq.shape[0])

        if not hasattr(obj, '_rng') or 'x0_sample_seed' in new_params:
            obj._rng = np.random.default_rng(
                getattr(obj, 'x0_sample_seed', None))

    def sample_initial_conditions(self, n_samples=1):
        x0 = self.parameters._rng.normal(size=(self.n_states, n_samples))
        if n_samples == 1:
            return x0.flatten()
        return x0

    def running_cost(self, x, u):
        if np.ndim(x) < 2:
            return np.zeros(1)

        return np.zeros(np.shape(x)[1])

    def dynamics(self, x, u):
        dxdt = np.matmul(self.parameters.freq, x)
        dxdt = [np.sin(dxdt[d]) for d in range(self.n_states)]
        dxdt = [np.sum(dxdt[d]) + u[d] for d in range(self.n_states)]
        dxdt = np.array(dxdt)

        if np.ndim(x) < 2:
            return dxdt.flatten()

        return dxdt


@pytest.mark.parametrize('n_states', (1, 2))
def test_SinusoidSystem(n_states):
    freq = np.eye(n_states)
    ocp = SinusoidSystem(freq=freq)
    controller = ConstantControl(np.zeros(ocp.n_controls))

    x = np.zeros(n_states)
    f = ocp.dynamics(x, controller(x))
    np.testing.assert_allclose(f, 0., atol=1e-14, rtol=1e-14)

    x = np.full((n_states,), np.pi/2.)
    f = ocp.dynamics(x, controller(x))
    np.testing.assert_allclose(f, 1., atol=1e-14, rtol=1e-14)

    x = np.full((n_states,), -np.pi/2.)
    f = ocp.dynamics(x, controller(x))
    np.testing.assert_allclose(f, -1., atol=1e-14, rtol=1e-14)

    x = np.full((n_states,), np.pi)
    f = ocp.dynamics(x, controller(x))
    np.testing.assert_allclose(f, 0., atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('mu', [-1., 1.])
@pytest.mark.parametrize('norm', (1, 2, np.inf))
@pytest.mark.parametrize('ftol', [1e-03, 1e-06])
@pytest.mark.parametrize('t_int', [-10., 10.])
def test_find_equilibrium_inside_limit_cycle(mu, norm, ftol, t_int):
    """Test that we can find stable and unstable equilibrium points with a guess
    inside the limit cycle."""
    ocp = VanDerPol(mu=mu)
    controller = ConstantControl(np.zeros(ocp.n_controls))

    # Initial guess
    x0 = ocp.sample_initial_conditions(distance=0.5)

    x = analyze.find_equilibrium(ocp, controller, x0, t_int, 10 * t_int,
                                 norm=norm, ftol=ftol)

    f = ocp.dynamics(x, controller(x))

    assert np.linalg.norm(f, ord=norm) < ftol
    np.testing.assert_allclose(x, 0., atol=ftol, rtol=ftol)


@pytest.mark.parametrize('mu', [-1., 1.])
@pytest.mark.parametrize('t_int', [-10., 10.])
def test_find_equilibrium_fails_outside_limit_cycle(mu, t_int):
    """Test that no equilibrium point is found with a guess outside the limit
    cycle, whether that limit cycle is stable or not."""
    ocp = VanDerPol(mu=mu)
    controller = ConstantControl(np.zeros(ocp.n_controls))

    # Initial guess
    x0 = ocp.sample_initial_conditions(distance=3.)

    with pytest.warns(RuntimeWarning, match="No equilibrium was found"):
        x, status = analyze.find_equilibrium(ocp, controller, x0, t_int,
                                             10 * t_int)

    # Integration should fail or reach the end of the integration horizon
    assert np.all(status != 0)

    # Double check that the point is not an equilibrium
    f = ocp.dynamics(x, controller(x))
    assert np.linalg.norm(f) >= 1e-03

    assert np.all(np.linalg.norm(x, axis=0) >= 1.)


@pytest.mark.parametrize('x0', (-2 * np.pi, -np.pi, 0., np.pi, 2 * np.pi))
def test_find_multiple_equilibria(x0):
    """
    For the system `dxdt = sin(x)`, we expect equilibria at integer multiples of
    pi.
    """
    ftol = 1e-03

    ocp = SinusoidSystem(freq=1.)
    controller = ConstantControl(np.zeros(ocp.n_controls))

    # Set the initial guess to be slightly closer to x0 than the next
    # equilibrium, x0 + pi
    x_guess = x0 + np.pi * 0.49

    x = analyze.find_equilibrium(ocp, controller, x_guess, 10., 100., ftol=ftol)

    # Since x_guess was closer to x0 than any other equilibrium, the result
    # should be equal to x0
    assert np.isclose(x, x0, atol=ftol, rtol=ftol)
