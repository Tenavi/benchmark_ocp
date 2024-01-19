import numpy as np
import pytest

from examples.uav.dynamics_model import containers
from examples.common_utilities.dynamics import (euler_to_quaternion,
                                                quaternion_to_euler)


def make_random_state(n_points, seed=None):
    rng = np.random.default_rng(seed)

    yaw = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))
    pitch = rng.uniform(low=-np.pi / 2., high=np.pi / 2., size=(n_points,))
    roll = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))

    return dict(pd=rng.normal(size=(n_points,)),
                u=rng.normal(size=(n_points,)),
                v=rng.normal(size=(n_points,)),
                w=rng.normal(size=(n_points,)),
                p=rng.normal(size=(n_points,)),
                q=rng.normal(size=(n_points,)),
                r=rng.normal(size=(n_points,)),
                attitude=np.squeeze(euler_to_quaternion([yaw, pitch, roll])))


@pytest.mark.parametrize('n_points', (1, 2, 11, 12))
def test_VehicleState_init(n_points):
    state_dict = make_random_state(n_points)

    container = containers.VehicleState(**state_dict.copy())

    assert container.n_points == n_points

    if n_points == 1:
        assert container.to_array().shape == (11,)
    else:
        assert container.to_array().shape == (11, n_points)

    for key, val in state_dict.items():
        np.testing.assert_array_equal(getattr(container, key), val)


def test_VehicleState_bad_size_init():
    pass
