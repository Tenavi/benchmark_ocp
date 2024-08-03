import numpy as np
import pytest

from optimalcontrol.controls import LinearQuadraticRegulator

from examples.common_utilities import dynamics, supervised_learning


rng = np.random.default_rng()


@pytest.mark.parametrize('n_points', [1, 2])
def test_angle_conversions(n_points):
    yaw = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))
    pitch = rng.uniform(low=-np.pi / 2., high=np.pi / 2., size=(n_points,))
    roll = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))

    c_yaw = np.cos(yaw / 2.)
    c_pitch = np.cos(pitch / 2.)
    c_roll = np.cos(roll / 2.)
    s_yaw = np.sin(yaw / 2.)
    s_pitch = np.sin(pitch / 2.)
    s_roll = np.sin(roll / 2.)

    q_expected = [c_yaw * c_pitch * s_roll - s_yaw * s_pitch * c_roll,
                  c_yaw * s_pitch * c_roll + s_yaw * c_pitch * s_roll,
                  s_yaw * c_pitch * c_roll - c_yaw * s_pitch * s_roll,
                  c_yaw * c_pitch * c_roll + s_yaw * s_pitch * s_roll]

    q = dynamics.euler_to_quaternion([yaw, pitch, roll])

    for i in range(4):
        np.testing.assert_allclose(q[i], q_expected[i], atol=1e-14)

    euler = dynamics.quaternion_to_euler(q)

    np.testing.assert_allclose(euler[0], yaw, atol=1e-14)
    np.testing.assert_allclose(euler[1], pitch, atol=1e-14)
    np.testing.assert_allclose(euler[2], roll, atol=1e-14)


@pytest.mark.parametrize('n_x', [1, 2])
@pytest.mark.parametrize('n_u', [1, 2])
def test_SimpleQRnet(n_x, n_u):
    xf = rng.normal(size=(n_x, 1)) / 100.
    uf = rng.normal(size=(n_u, 1)) / 100.

    u_lb = -1.
    u_ub = 1.

    A = rng.normal(size=(n_x, n_x))
    B = rng.normal(size=(n_x, n_u))
    Q = np.identity(n_x)
    R = np.identity(n_u)

    lqr = LinearQuadraticRegulator(A, B, Q, R, xf=xf, uf=uf,
                                   u_lb=u_lb, u_ub=u_ub)

    ctrl = supervised_learning.SimpleQRnet(
        lqr, supervised_learning.PolynomialController, degree=3)

    np.testing.assert_array_equal(ctrl.u_lb, u_lb)
    np.testing.assert_array_equal(ctrl.u_ub, u_ub)

    # Generate linear training data using lqr
    x_train = rng.normal(xf, size=(n_x, 10))
    x_test = rng.normal(xf, size=(n_x, 10))
    u_train = lqr(x_train)

    ctrl.train(x_train, u_train)

    assert ctrl.n_states == n_x
    assert ctrl.n_controls == n_u
    assert ctrl.train_time is ctrl.wrapped_controller.train_time

    # Since lqr exactly equals the data, the polynomial part should do nothing
    np.testing.assert_allclose(ctrl._wrapped_uf, 0., atol=1e-14)

    np.testing.assert_allclose(ctrl(x_test), lqr(x_test), rtol=1e-10)

    # Now modify the data to have nonlinearities and retrain
    k2, k3 = 0.1, -0.01
    u_train = lqr(x_train) + k2 * x_train ** 2 + k3 * x_train ** 3

    ctrl.train(x_train, u_train)

    # There should be some differences between lqr and the new controls
    assert np.abs(ctrl(x_test) - lqr(x_test)).max() > 0.01
