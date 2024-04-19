import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from examples.common_utilities.dynamics import (euler_to_quaternion,
                                                quaternion_to_euler)
from examples.uav.fixed_wing_dynamics import containers
from examples.uav.vehicle_models.aerosonde import constants


def rotation_matrix(yaw, pitch, roll):
    n = np.size(yaw)
    yaw_mat = np.zeros((3, 3, n))
    pitch_mat = np.zeros((3, 3, n))
    roll_mat = np.zeros((3, 3, n))

    yaw_mat[0, 0] = np.cos(yaw)
    yaw_mat[0, 1] = np.sin(yaw)
    yaw_mat[1, 0] = -yaw_mat[0, 1]
    yaw_mat[1, 1] = yaw_mat[0, 0]
    yaw_mat[2, 2] = 1.

    pitch_mat[0, 0] = np.cos(pitch)
    pitch_mat[0, 2] = -np.sin(pitch)
    pitch_mat[1, 1] = 1.
    pitch_mat[2, 0] = -pitch_mat[0, 2]
    pitch_mat[2, 2] = pitch_mat[0, 0]

    roll_mat[0, 0] = 1.
    roll_mat[1, 1] = np.cos(roll)
    roll_mat[1, 2] = np.sin(roll)
    roll_mat[2, 1] = -roll_mat[1, 2]
    roll_mat[2, 2] = roll_mat[1, 1]

    rot_mat = np.einsum('ij...,jk...->ik...', pitch_mat, yaw_mat)
    rot_mat = np.einsum('ij...,jk...->ik...', roll_mat, rot_mat)

    return rot_mat


def random_attitude(n_points, seed=None):
    rng = np.random.default_rng(seed)

    yaw = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))
    pitch = rng.uniform(low=-np.pi / 2., high=np.pi / 2., size=(n_points,))
    roll = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))

    return yaw, pitch, roll


def random_states(n_points, seed=None):
    state_dict, _ = _random_state_array(n_points, seed)
    return containers.VehicleState(**state_dict)


def random_controls(n_points, seed=None):
    control_dict, _ = _random_control_array(n_points, seed)
    return containers.Controls(**control_dict)


def _random_state_array(n_points, seed=None):
    rng = np.random.default_rng(seed)

    yaw, pitch, roll = random_attitude(n_points, rng)
    attitude = euler_to_quaternion([yaw, pitch, roll])

    state_array = np.vstack([rng.normal(size=(7, n_points)), attitude])

    state_dict = dict(pd=state_array[0],
                      u=state_array[1],
                      v=state_array[2],
                      w=state_array[3],
                      p=state_array[4],
                      q=state_array[5],
                      r=state_array[6],
                      attitude=np.squeeze(attitude))

    return state_dict, np.squeeze(state_array)


def _random_control_array(n_points, seed=None):
    rng = np.random.default_rng(seed)

    control_array = np.vstack([rng.uniform(size=(1, n_points)),
                               rng.uniform(low=-25. * np.pi / 180.,
                                           high=25. * np.pi / 180.,
                                           size=(3, n_points))])

    control_dict = dict(throttle=control_array[0],
                        aileron=control_array[1],
                        elevator=control_array[2],
                        rudder=control_array[3])

    return control_dict, np.squeeze(control_array)


def assert_container_equal(container, array, attr_dict):
    """Confirm that a given container matches an expected dict and array."""
    for key, val in attr_dict.items():
        np.testing.assert_array_equal(getattr(container, key), val)
    np.testing.assert_array_equal(array, container.to_array())


def assert_attr_update_equal(container, update_dict):
    orig_array = container.to_array(copy=True)

    for idx, attr in enumerate(update_dict.keys()):
        # First check that previous updates haven't changed subsequent states
        np.testing.assert_array_equal(getattr(container, attr),
                                      orig_array[idx])
        np.testing.assert_array_equal(container.to_array()[idx],
                                      orig_array[idx])

        # Update the attribute and check it's updated
        setattr(container, attr, update_dict[attr])

        np.testing.assert_array_equal(getattr(container, attr),
                                      update_dict[attr])
        np.testing.assert_array_equal(container.to_array()[idx],
                                      update_dict[attr])


@pytest.mark.parametrize('n_points', [1, 2])
def test_angle_conversions(n_points):
    yaw, pitch, roll = random_attitude(n_points)

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

    q = euler_to_quaternion([yaw, pitch, roll])

    for i in range(4):
        np.testing.assert_allclose(q[i], q_expected[i], atol=1e-14)

    euler = quaternion_to_euler(q)

    np.testing.assert_allclose(euler[0], yaw, atol=1e-14)
    np.testing.assert_allclose(euler[1], pitch, atol=1e-14)
    np.testing.assert_allclose(euler[2], roll, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_VehicleState_init(n_points):
    state_dict, state_array = _random_state_array(n_points)
    container = containers.VehicleState(**state_dict)

    assert container._array.shape == (11, n_points)
    if n_points == 1:
        assert container.to_array().shape == (11,)
    else:
        assert container.to_array().shape == (11, n_points)

    assert_container_equal(container, state_array, state_dict)

    # Initialized from array
    for array in [state_array, state_array.reshape(11, -1)]:
        container = containers.VehicleState.from_array(array)
        assert container._array.shape == (11, n_points)

    assert_container_equal(container, state_array, state_dict)


@pytest.mark.parametrize('n_points', [1, 2])
def test_Controls_init(n_points):
    ctrl_dict, ctrl_array = _random_control_array(n_points)

    container = containers.Controls(**ctrl_dict)

    assert container._array.shape == (4, n_points)
    if n_points == 1:
        assert container.to_array().shape == (4,)
    else:
        assert container.to_array().shape == (4, n_points)

    assert_container_equal(container, ctrl_array, ctrl_dict)

    # Initialized from array
    for array in [ctrl_array, ctrl_array.reshape(4, -1)]:
        container = containers.Controls.from_array(array)
        assert container._array.shape == (4, n_points)

    assert_container_equal(container, ctrl_array, ctrl_dict)


@pytest.mark.parametrize('n_points', [1, 2])
def test_VehicleState_update(n_points):
    state_dict, state_array = _random_state_array(n_points)
    state_dict2, state_array2 = _random_state_array(n_points)

    assert not np.any(state_array == state_array2)

    container = containers.VehicleState(**state_dict)

    assert_container_equal(container, state_array, state_dict)

    new_attitude = state_dict2.pop('attitude')

    assert_attr_update_equal(container, state_dict2)

    state_dict2['attitude'] = new_attitude

    container.attitude = state_dict2['attitude']
    np.testing.assert_array_equal(container.attitude, state_array2[7:])
    np.testing.assert_array_equal(container.to_array()[7:], state_array2[7:])

    assert_container_equal(container, state_array2, state_dict2)


@pytest.mark.parametrize('n_points', [1, 2])
def test_Controls_update(n_points):
    ctrl_dict, ctrl_array = _random_control_array(n_points)
    ctrl_dict2, ctrl_array2 = _random_control_array(n_points)

    assert not np.any(ctrl_array == ctrl_array2)

    container = containers.Controls(**ctrl_dict)

    assert_container_equal(container, ctrl_array, ctrl_dict)

    assert_attr_update_equal(container, ctrl_dict2)

    assert_container_equal(container, ctrl_array2, ctrl_dict2)


@pytest.mark.parametrize('n_points', [1, 2])
@pytest.mark.parametrize('update_attr', ['u', 'v', 'w'])
def test_airspeed_update(n_points, update_attr):
    container = random_states(n_points)

    # Initialize _airspeed attribute
    assert container._airspeed is None
    _ = container.airspeed
    assert container._airspeed is not None

    # Update velocity state
    new_velocity = getattr(container, update_attr) * 10.

    setattr(container, update_attr, new_velocity)

    # Check that velocity has been updated and _airspeed has been reset
    np.testing.assert_array_equal(getattr(container, update_attr), new_velocity)
    assert container._airspeed is None

    # Compute comparison airspeed
    u, v, w = container.u, container.v, container.w

    Va_expect = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    alpha_expect = np.arctan2(w, u)
    beta_expect = np.arcsin(v / Va_expect)

    Va, alpha, beta = container.airspeed

    np.testing.assert_allclose(Va, Va_expect, atol=1e-14)
    np.testing.assert_allclose(alpha, alpha_expect, atol=1e-14)
    np.testing.assert_allclose(beta, beta_expect, atol=1e-14)

    # Check that _airspeed gets updated
    assert container._airspeed.shape == (3, n_points)
    assert container.airspeed is container._airspeed


@pytest.mark.parametrize('n_points', [1, 2])
def test_zero_airspeed(n_points):
    state_dict, _ = _random_state_array(n_points)

    # Set the first airspeed states to zero
    state_dict['u'][0] = 0.
    state_dict['v'][0] = 0.
    state_dict['w'][0] = 0.

    container = containers.VehicleState(**state_dict)

    u, v, w = container.u, container.v, container.w

    Va_expect = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    alpha_expect = np.arctan2(w, u)
    beta_expect = np.zeros(n_points)
    beta_expect[1:] = np.arcsin(v[1:] / Va_expect[1:])

    Va, alpha, beta = container.airspeed

    np.testing.assert_allclose(Va, Va_expect, atol=1e-14)
    np.testing.assert_allclose(alpha, alpha_expect, atol=1e-14)
    np.testing.assert_allclose(beta, beta_expect, atol=1e-14)

    np.testing.assert_array_equal(container.airspeed[:, 0], 0.)


@pytest.mark.parametrize('n_points', [1, 2])
def test_rotation(n_points):
    container = random_states(n_points)

    yaw, pitch, roll = quaternion_to_euler(container.attitude)
    rot_mat = rotation_matrix(yaw, pitch, roll)

    # Make random vectors and rotate to body frame
    vec = np.random.default_rng().normal(size=(3, n_points))
    vec_rot = container.inertial_to_body(vec)

    assert vec_rot.shape == vec.shape

    for i in range(n_points):
        vec_rot_expected = rot_mat[..., i] @ vec[:, i]
        np.testing.assert_allclose(vec_rot[:, i], vec_rot_expected, atol=1e-14)

    # Make random vectors and rotate to inertial frame
    vec = np.random.default_rng().normal(size=(3, n_points))
    vec_rot = container.body_to_inertial(vec)

    assert vec_rot.shape == vec.shape

    for i in range(n_points):
        vec_rot_expected = np.linalg.solve(rot_mat[..., i], vec[:, i])
        np.testing.assert_allclose(vec_rot[:, i], vec_rot_expected, atol=1e-14)


@pytest.mark.parametrize('fun', ['inertial_to_body', 'body_to_inertial'])
def test_rotation_shape(fun):
    # Make two rotations and containers for each and both
    _, state_array = _random_state_array(2)
    state_array[:7] = 0.

    cont1 = containers.VehicleState.from_array(state_array[:, 0])
    cont2 = containers.VehicleState.from_array(state_array[:, 1])
    cont = containers.VehicleState.from_array(state_array)

    # Make a single random vector and rotate
    vec = np.random.default_rng().normal(size=(3,))
    vec_rot1 = getattr(cont1, fun)(vec)
    vec_rot2 = getattr(cont2, fun)(vec)
    vec_rot12 = getattr(cont, fun)(vec)

    assert vec_rot1.shape == vec_rot2.shape == (3,)
    assert vec_rot12.shape == (3, 2)

    np.testing.assert_allclose(vec_rot1, vec_rot12[:, 0], atol=1e-14)
    np.testing.assert_allclose(vec_rot2, vec_rot12[:, 1], atol=1e-14)

    # Make a single random vector with shape (3, 1) and rotate
    vec = np.random.default_rng().normal(size=(3, 1))
    vec_rot1 = getattr(cont1, fun)(vec)
    vec_rot2 = getattr(cont2, fun)(vec)
    vec_rot12 = getattr(cont, fun)(vec)

    assert vec_rot1.shape == vec_rot2.shape == (3, 1)
    assert vec_rot12.shape == (3, 2)

    np.testing.assert_allclose(vec_rot1, vec_rot12[:, :1], atol=1e-14)
    np.testing.assert_allclose(vec_rot2, vec_rot12[:, 1:], atol=1e-14)

    # Make two random vectors and rotate
    vec = np.random.default_rng().normal(size=(3, 2))
    vec_rot1 = getattr(cont1, fun)(vec)
    vec_rot2 = getattr(cont2, fun)(vec)
    vec_rot12 = getattr(cont, fun)(vec)

    assert vec_rot1.shape == vec_rot2.shape == vec_rot12.shape == (3, 2)

    # The state with two rotations broadcasts the rotations on each vector
    np.testing.assert_allclose(vec_rot1[:, 0], vec_rot12[:, 0], atol=1e-14)
    np.testing.assert_allclose(vec_rot2[:, 1], vec_rot12[:, 1], atol=1e-14)
    # The other vectors are rotated differently by each state
    np.testing.assert_allclose(vec_rot1[:, 1], getattr(cont1, fun)(vec[:, 1]),
                               atol=1e-14)
    np.testing.assert_allclose(vec_rot2[:, 0], getattr(cont2, fun)(vec[:, 0]),
                               atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_rotation_update(n_points):
    container = random_states(n_points)

    # Initialize _rotation attribute
    assert container._rotation is None
    assert isinstance(container.rotation, Rotation)
    assert container.rotation is container._rotation

    # Update attitude
    yaw, pitch, roll = random_attitude(n_points)
    quat = euler_to_quaternion([yaw, pitch, roll])
    container.attitude = quat

    # Check that attitude has been updated and _rotation has been reset
    np.testing.assert_array_equal(container.attitude, np.squeeze(quat))
    assert container._rotation is None

    # Make random vectors and rotate to body frame
    vec = np.random.default_rng().normal(size=(3, n_points))
    vec_rot = container.inertial_to_body(vec)

    # Rotate back to inertial frame and check that we recover the initial vector
    vec_rot = container.body_to_inertial(vec_rot)
    np.testing.assert_allclose(vec_rot, vec, atol=1e-14)

    assert isinstance(container.rotation, Rotation)
    assert container.rotation is container._rotation


@pytest.mark.parametrize('n_points', [1, 2])
@pytest.mark.parametrize('update_attr', ['u', 'v', 'w', 'attitude'])
def test_course_update(n_points, update_attr):
    container = random_states(n_points)

    # Initialize _course attribute
    assert container._course is None
    _ = container.course
    assert container._course is not None

    # Update attitude or velocity
    if 'update_attr' == 'attitude':
        yaw, pitch, roll = random_attitude(n_points)
        quat = euler_to_quaternion([yaw, pitch, roll])
        container.attitude = quat
    else:
        new_velocity = getattr(container, update_attr) * -10.
        setattr(container, update_attr, new_velocity)

    # Check that _course has been reset
    assert container._course is None

    # Compute comparison course
    u, v, w = container.u, container.v, container.w
    u_in, v_in, _ = container.body_to_inertial([u, v, w])
    course_expect = np.arctan2(v_in, u_in)

    np.testing.assert_allclose(container.course, course_expect, atol=1e-14)

    # Check that _course gets updated
    assert container.course is container._course


@pytest.mark.parametrize('n_points', [1, 2])
@pytest.mark.parametrize('multiplier', [-1., 1/2, 1.])
@pytest.mark.parametrize('inplace', [True, False])
def test_saturate(n_points, multiplier, inplace):
    rng = np.random.default_rng()

    ctrl_array = np.vstack([rng.uniform(low=1., high=2., size=(1, n_points)),
                            rng.uniform(low=25. * np.pi / 180.,
                                        high=50. * np.pi / 180.,
                                        size=(3, n_points))])
    ctrl_array *= multiplier

    unsat_container = containers.Controls.from_array(ctrl_array)

    container = unsat_container.saturate(constants.min_controls,
                                         constants.max_controls,
                                         inplace=inplace)
    if inplace:
        assert container is unsat_container
    else:
        assert container is not unsat_container
        np.testing.assert_array_equal(unsat_container.to_array(),
                                      np.squeeze(ctrl_array))

    sat_ctrl_array = container.to_array().reshape(ctrl_array.shape)

    for i in range(n_points):
        if multiplier == -1.:
            np.testing.assert_array_equal(sat_ctrl_array[:, i],
                                          constants.min_controls.to_array())
        elif multiplier == 1/2:
            np.testing.assert_array_equal(sat_ctrl_array[:, i],
                                          ctrl_array[:, i])
        else:
            np.testing.assert_array_equal(sat_ctrl_array[:, i],
                                          constants.max_controls.to_array())
