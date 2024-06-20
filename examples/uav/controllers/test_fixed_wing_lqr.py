import numpy as np
import pytest

from optimalcontrol import analyze
from optimalcontrol.simulate import integrate

from tests._utilities import compare_finite_difference

from examples.uav import FixedWing
from examples.uav.controllers import FixedWingLQR
from examples.uav import example_config as config


@pytest.mark.parametrize('n_points', [1, 2])
def test_lqr_jac(n_points):
    ocp = FixedWing(**config.params)
    lqr = FixedWingLQR(ocp)

    # Test at single point
    x = ocp.sample_initial_conditions(n_points)

    dudx = lqr.jac(x)

    if n_points == 1:
        assert dudx.shape == (ocp.n_controls, ocp.n_states)
    else:
        assert dudx.shape == (ocp.n_controls, ocp.n_states, n_points)

    compare_finite_difference(x, dudx, lqr, method='3-point')


def test_locally_stable():
    ocp = FixedWing(**config.params)
    lqr = FixedWingLQR(ocp)

    # Expected trim states and controls
    x_trim = ocp.trim_state
    u_trim = ocp.trim_controls

    # Find actual trim state and control by integration
    xf, status = analyze.find_equilibrium(ocp, lqr, x_trim, config.t_int,
                                          config.t_max)

    assert np.sum(status == 0) == 1

    # This should be reasonably close to expected
    np.testing.assert_allclose(xf, x_trim, atol=1e-03, rtol=1e-05)
    np.testing.assert_allclose(lqr(xf), u_trim, atol=1e-03, rtol=1e-05)

    # LQR should be linearly stable at trim
    _, _, max_eig = analyze.linear_stability(ocp, lqr, xf)
    assert np.real(max_eig) < 0.

    # Verify that a small perturbation from trim returns to trim.
    x0 = ocp.sample_initial_conditions(distance=0.1).reshape(-1, 1)

    t, x, status = integrate(ocp, lqr, x0, [0., config.t_int],
                             **config.sim_kwargs)

    assert status == 0
    np.testing.assert_allclose(x[:, -1], xf, atol=1e-03)
