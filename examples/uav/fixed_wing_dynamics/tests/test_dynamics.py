import numpy as np
import pytest

from examples.common_utilities.dynamics import quaternion_to_euler
from examples.uav.fixed_wing_dynamics import dynamics
from examples.uav.vehicle_models.aerosonde import constants, aero_model

from .test_containers import random_states, random_controls


rng = np.random.default_rng()


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_position(n_points):
    states = random_states(n_points)

    _, pitch, roll = quaternion_to_euler(states.attitude)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    d_pd_expect = (-np.sin(pitch) * states.u
                   + np.sin(roll) * np.cos(pitch) * states.v
                   + np.cos(roll) * np.cos(pitch) * states.w)

    np.testing.assert_allclose(dxdt.pd, d_pd_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_velocity(n_points):
    states = random_states(n_points)

    forces = rng.normal(size=(3, n_points))
    moments = rng.normal(size=(3, n_points))

    dxdt = dynamics.rigid_body_dynamics(states, forces, moments, constants)

    # Brute force calculation from Beard (3.7)
    d_vb_expect = np.stack([states.r * states.v - states.q * states.w,
                            states.p * states.w - states.r * states.u,
                            states.q * states.u - states.p * states.v],
                           axis=0)
    d_vb_expect = np.squeeze(d_vb_expect + forces / constants.mass)

    np.testing.assert_allclose(dxdt.velocity, d_vb_expect, atol=1e-14)


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
def test_gravity(n_points):
    states = random_states(n_points)

    _, pitch, roll = quaternion_to_euler(states.attitude)

    g_expect = [-np.sin(pitch),
                np.cos(pitch) * np.sin(roll),
                np.cos(pitch) * np.cos(roll)]
    g_expect = np.stack(g_expect, axis=0)
    g_expect = constants.mass * constants.g0 * g_expect

    if n_points == 1:
        assert g_expect.shape == (3,)
    else:
        assert g_expect.shape == (3, n_points)

    gravity = dynamics.gravity(states, constants.mg)

    np.testing.assert_allclose(gravity, g_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_dynamics_shapes(n_points):
    states = random_states(n_points)
    controls = random_controls(n_points)

    dxdt = dynamics.dynamics(states, controls, constants, aero_model)

    assert dxdt.to_array().shape == states.to_array().shape
