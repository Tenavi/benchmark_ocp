import numpy as np
import pytest

from optimalcontrol.utilities import approx_derivative

from examples.common_utilities.dynamics import (euler_to_quaternion,
                                                quaternion_to_euler)

from examples.fixed_wing.fixed_wing_dynamics import dynamics
from examples.fixed_wing.fixed_wing_dynamics.containers import VehicleState, Controls
from examples.fixed_wing.vehicle_models.aerosonde import constants, aero_model

from .test_containers import random_states, random_controls


rng = np.random.default_rng()


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_position(n_points):
    states = random_states(n_points)

    _, pitch, roll = quaternion_to_euler(states.attitude)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    d_h_expect = -(-np.sin(pitch) * states.u
                   + np.sin(roll) * np.cos(pitch) * states.v
                   + np.cos(roll) * np.cos(pitch) * states.w)

    np.testing.assert_allclose(dxdt.h, d_h_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_velocity(n_points):
    states = random_states(n_points)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    angles = quaternion_to_euler(states.attitude).reshape(3, n_points)
    pitch, roll = angles[1:]
    gravity = [-np.sin(pitch),
               np.cos(pitch) * np.sin(roll),
               np.cos(pitch) * np.cos(roll)]
    gravity = constants.mass * constants.g0 * np.stack(gravity, axis=0)

    # Brute force calculation from Beard (3.7)
    d_vb_expect = np.stack([states.r * states.v - states.q * states.w,
                            states.p * states.w - states.r * states.u,
                            states.q * states.u - states.p * states.v],
                           axis=0)
    d_vb_expect = np.squeeze(d_vb_expect + (forces + gravity) / constants.mass)

    np.testing.assert_allclose(dxdt.velocity, d_vb_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_rates(n_points):
    states = random_states(n_points)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    p, q, r = states.rates
    Jx, Jy, Jz, Jxz = constants.Jxx, constants.Jyy, constants.Jzz, constants.Jxz

    d_omega_expect = [Jxz * p * q + (Jy - Jz) * q * r,
                      Jxz * (r ** 2 - p ** 2) + (Jz - Jx) * p * r,
                      (Jx - Jy) * p * q - Jxz * q * r]
    d_omega_expect = np.stack(d_omega_expect, axis=0) + np.squeeze(moments)
    d_omega_expect = constants.J_inv_body @ d_omega_expect

    np.testing.assert_allclose(dxdt.rates, d_omega_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_quaternion(n_points):
    states = random_states(n_points)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    quat = states.attitude.reshape(4, -1)
    d_quat_expect = np.empty_like(quat)

    for i in range(n_points):
        Q = [[0., states.r[i], -states.q[i], states.p[i]],
             [-states.r[i], 0., states.p[i], states.q[i]],
             [states.q[i], -states.p[i], 0., states.r[i]],
             [-states.p[i], -states.q[i], -states.r[i], 0.]]
        d_quat_expect[:, i] = 0.5 * np.matmul(Q, quat[:, i])
    d_quat_expect = np.squeeze(d_quat_expect)

    np.testing.assert_allclose(dxdt.attitude, d_quat_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_axis_rates(n_points):
    """
    Verify that when roll and pitch are zero, body angular rates equal Euler
    angle rates, i.e. [p, q, r] = d/dt [phi, theta, psi].
    """
    dt = 1e-07
    tol = 10. * dt

    rates = np.squeeze(rng.normal(scale=np.pi / 180., size=(3, n_points)))

    # Any yaw angle should be okay
    angles = np.zeros((3, n_points))
    angles[0] = rng.uniform(low=-np.pi, high=np.pi, size=(1, n_points))
    angles = np.squeeze(angles)
    attitude = euler_to_quaternion(np.squeeze(angles))

    state = VehicleState(attitude=attitude, **dict(zip(('p', 'q', 'r'), rates)))

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    # Finite different approximation of Euler angle dynamics (not equivalent to
    # converting dxdt.attitude to Euler angles!)
    dxdt = dynamics.rigid_body_dynamics(state, forces, moments, constants)
    new_state = state + dt * dxdt

    new_angles = quaternion_to_euler(new_state.attitude, degrees=False)

    d_angles = (new_angles - angles) / dt

    # Angles are in [yaw, pitch, roll] order for euler_to_quaternion, but rates
    # are in [p, q, r] order, so look at d_angles in reverse.
    np.testing.assert_allclose(d_angles[::-1], rates, atol=tol, rtol=tol)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_shapes(n_points):
    states = random_states(n_points)
    controls = random_controls(n_points)

    dxdt = dynamics.dynamics(states, controls, constants, aero_model)

    assert dxdt.to_array().shape == states.to_array().shape


@pytest.mark.parametrize('n_points', [1, 2])
def test_rigid_body_jac(n_points):
    """Test the rigid body components of the Jacobians, i.e. without
    contributions from aero-propulsive forces and moments."""
    tol = dict(rtol=1e-06, atol=1e-10)

    states = random_states(n_points)
    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    # We leave forces and moments constant when evaluating the dynamics for
    # finite differencing, so the force and moment Jacobians are artificially 0.
    zero_jac = np.zeros((3, 6, n_points))

    dfdx = dynamics.rigid_body_jac(states, zero_jac, zero_jac, constants)

    if n_points == 1:
        assert dfdx.shape == (11, 11)
    else:
        assert dfdx.shape == (11, 11, n_points)

    def dynamics_wrapper(x):
        state = VehicleState.from_array(x)
        dxdt = dynamics.rigid_body_dynamics(state, forces, moments, constants)
        return dxdt.to_array()

    dfdx_expect = approx_derivative(dynamics_wrapper, states.to_array())

    # Altitude
    np.testing.assert_allclose(dfdx[0], dfdx_expect[0], **tol)

    # Velocity
    np.testing.assert_allclose(dfdx[1:4], dfdx_expect[1:4], **tol)

    # Rates
    np.testing.assert_allclose(dfdx[4:7], dfdx_expect[4:7], **tol)

    # Quaternions
    np.testing.assert_allclose(dfdx[7:], dfdx_expect[7:], **tol)


@pytest.mark.parametrize('n_points', [1, 2])
def test_jacobians(n_points):
    tol = dict(rtol=1e-06, atol=1e-10)

    states = random_states(n_points)
    controls = random_controls(n_points)

    dfdx, dfdu = dynamics.jacobians(states, controls, constants, aero_model)

    if n_points == 1:
        assert dfdx.shape == (11, 11)
        assert dfdu.shape == (11, 4)
    else:
        assert dfdx.shape == (11, 11, n_points)
        assert dfdu.shape == (11, 4, n_points)

    def dynamics_wrapper(x, u):
        if not isinstance(x, VehicleState):
            x = VehicleState.from_array(x)
        if not isinstance(u, Controls):
            u = Controls.from_array(u)
        dxdt = dynamics.dynamics(x, u, constants, aero_model)
        return dxdt.to_array()

    dfdx_expect = approx_derivative(lambda x: dynamics_wrapper(x, controls),
                                    states.to_array())
    dfdu_expect = approx_derivative(lambda u: dynamics_wrapper(states, u),
                                    controls.to_array())

    np.testing.assert_allclose(dfdx, dfdx_expect, **tol)
    np.testing.assert_allclose(dfdu, dfdu_expect, **tol)
