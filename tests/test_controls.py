import numpy as np
import pytest

from optimalcontrol import controls

from tests._utilities import make_LQ_params


rng = np.random.default_rng()


@pytest.mark.parametrize('n_states', range(1, 4))
@pytest.mark.parametrize('n_controls', range(1, 4))
def test_ConstantControl(n_states, n_controls):
    u = rng.normal(size=(n_controls, 1))

    ctrl = controls.ConstantControl(u)

    assert ctrl.n_controls == n_controls
    np.testing.assert_array_equal(u, ctrl.u)

    x = rng.normal(size=(n_states,))
    u = ctrl(x)
    dudx = ctrl.jac(x)
    assert u.shape == (n_controls,)
    assert dudx.shape == (n_controls, n_states)
    np.testing.assert_array_equal(u, ctrl.u.flatten())
    np.testing.assert_array_equal(dudx, 0.)

    for n_points in (1, 2, 3):
        x = rng.normal(size=(n_states, n_points))
        u = ctrl(x)
        dudx = ctrl.jac(x)
        assert u.shape == (n_controls, n_points)
        assert dudx.shape == (n_controls, n_states, n_points)
        np.testing.assert_array_equal(dudx, 0.)

        for i in range(n_points):
            np.testing.assert_array_equal(u[:, i:i + 1], ctrl.u)


@pytest.mark.parametrize('zero_index', (0, 1, 2, [0, 1], [0, 2], [1, 2]))
def test_zero_column_lqr(zero_index):
    """
    Test that LQR can be created when one or more columns of A and Q are zero.
    This creates a situation where some states don't impact dynamics of other
    states or the cost function. The Riccati solver will often fail for the full
    set of states, but can find a solution to the sub-problem which ignores
    these states. Using the resulting control law in the full system stabilizes
    the controlled states, with the ignored states' stability depending on their
    dynamics. Since such problems do appear in practice (e.g. linearized rigid
    body attitude control with quaternion attitude representation), it is
    desirable to be able to generate an LQR controller in these situations.
    """
    n_states = 3
    n_controls = 2

    # Start with usual random matrices
    A, B, Q, R, _, _ = make_LQ_params(n_states, n_controls)

    # Set some columns of A and Q to zero
    A[:, zero_index] = 0.
    Q[zero_index] = 0.
    Q[:, zero_index] = 0.

    # Should still be able to make an lqr controller
    lqr = controls.LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)

    # The closed-loop eigenvalues should be non-positive
    A_cl = A + np.matmul(B, - lqr.K)
    assert np.linalg.eigvals(A_cl).real.max() <= 0.
