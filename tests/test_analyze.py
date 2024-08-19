import numpy as np
import pytest

from optimalcontrol import analyze
from optimalcontrol.problem import OptimalControlProblem, LinearQuadraticProblem
from optimalcontrol.controls import ConstantControl, LinearQuadraticRegulator

from examples.van_der_pol import VanDerPol


rng = np.random.default_rng()


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

    x, status = analyze.find_equilibrium(ocp, controller, x0, t_int,
                                         10. * t_int, norm=norm, ftol=ftol)

    assert np.sum(status == 0) == 1

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
                                             10. * t_int)

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

    x, status = analyze.find_equilibrium(ocp, controller, x_guess, 10., 100.,
                                         ftol=ftol)

    assert np.sum(status == 0) == 1
    assert np.sum(status) == 3

    # Since x_guess was closer to x0 than any other equilibrium, the result
    # should be equal to x0
    assert np.isclose(x, x0, atol=ftol, rtol=ftol)


@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('n_w', [1, 2, 3])
def test_scale_matrix(n, n_w):
    d = rng.normal(loc=1., scale=0.1, size=(n_w, n))
    M = rng.normal(size=(n_w, n, n)) + 1j * rng.normal(size=(n_w, n, n))

    dMd = analyze.robustness._scale_matrix(d, M)

    assert dMd.shape == M.shape

    for i in range(n_w):
        D = np.diag(d[i])
        dMd_expect = D @ M[i] @ np.linalg.inv(D)
        np.testing.assert_allclose(dMd[i], dMd_expect, atol=1e-12)


def test_disk_margins_siso():
    r"""
    System from example 1 in Seiler et al. (2020):
        $P(s) = Y(s) / U(s) = 1 / (s^3 + 10s^2 + 10s + 10)$
        $K(s) = 25$
    Setting $x_1 = y$, we can write this in state space form as
        $dx_1/dt = x_2$
        $dx_2/dt = x_3$
        $dx_3/dt = -10 (x_1 + x_2 + x_3) + u$
        $u = -25 x_1$
    """
    A = np.array([[0., 1., 0.],
                  [0., 0., 1.],
                  [-10., -10., -10.]])
    B = np.array([[0.],
                  [0.],
                  [1.]])
    C = np.array([[1., 0., 0.]])
    K = 25. * C
    Q = C.T @ C
    R = np.ones((1, 1))

    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-10., x0_ub=10.)
    ctrl = LinearQuadraticRegulator(K=K)

    # Make sure the state space system is set up correctly
    _, eigs, _ = analyze.linear_stability(ocp, ctrl, ctrl.xf, verbose=False)

    np.testing.assert_allclose(eigs.real, [-9.33, -0.33, -0.33], atol=0.01)
    np.testing.assert_allclose(np.abs(eigs.imag), [0., 1.91, 1.91], atol=0.01)

    margins = analyze.disk_margins(ocp, ctrl, ctrl.xf)

    np.testing.assert_allclose(margins['disk_margin'], 0.46, atol=0.01)
    np.testing.assert_allclose(margins['critical_frequency'], 1.94, atol=0.05)

    gm = margins['gain_margin']
    pm_expect = (1. + gm.prod()) / gm.sum()
    pm_expect = np.rad2deg(np.arccos(pm_expect)) * np.array([-1., 1.])

    np.testing.assert_allclose(gm, [0.63, 1.59], atol=0.01)
    np.testing.assert_allclose(margins['phase_margin'], pm_expect, atol=1e-12)


def test_disk_margins_mimo():
    A = np.array([[-0.2529, 0.6962, -1.9870, -9.7491, 0],
                  [-0.6108, -3.6183, 19.4199, -0.9979, 0],
                  [0.3036, -2.9669, -4.2358, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0.1018, -0.9948, 0, 20, 0]])
    B = np.array([[-0.0025, 5.3843],
                  [-1.6575, 0],
                  [-23.1119, 0],
                  [0, 0],
                  [0, 0]])
    Q = np.diag([1.,
                 1.,
                 np.deg2rad(30.) ** -2,
                 np.deg2rad(5.) ** -2,
                 100. ** -2])
    R = np.diag([np.deg2rad(30.) ** -2,
                 1.])

    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-10., x0_ub=10.)
    ctrl = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)

    margins = analyze.disk_margins(ocp, ctrl, ctrl.xf, tol=1e-10)

    np.testing.assert_allclose(margins['disk_margin'], 1.8633, atol=0.01)
    np.testing.assert_allclose(margins['critical_frequency'], 27.1031, atol=0.5)

    gm_expect = [0.0354, 28.2696]
    pm_expect = [-85.9482, 85.9482]

    np.testing.assert_allclose(margins['gain_margin'], gm_expect, rtol=0.05)
    np.testing.assert_allclose(margins['phase_margin'], pm_expect, rtol=0.05)
