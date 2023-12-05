import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def cross_product_matrix(w):
    """
    Construct the cross product matrix or matrices from one or more vectors.

    Parameters
    ----------
    w : (3, n_vectors) or (3,) array
        Vector(s) to construct cross product matrices for.

    Returns
    -------
    w_x : (3, 3, n_vectors) or (3, 3) array
        If `w` is a 1d array, then `w_x` is a 2d array with entries
        ```
        w_x = [[ 0,    -w[2], w[1]  ],
               [ w[2], 0,     -w[0] ],
               [ -w[1], w[0], 0     ]]
        ```
        If `w` is a 2d array, then `w_x[:, :, i]` is the cross product matrix
        (see above) for `w[:, i]`.
    """
    zeros = np.zeros_like(w[0])
    return np.array([[zeros, -w[2], w[1]],
                     [w[2], zeros, -w[0]],
                     [-w[1], w[0], zeros]])


def quaternion_to_euler(quat, degrees=False, normalize=True,
                        ignore_warnings=True):
    """
    Convert angles in quaternion representation to Euler angles.

    Parameters
    ----------
    quat : (4, n_angles) or (4,) array
        Angles in quaternion representation. `quat[:3]` are assumed to contain
        the vector portion of the quaternion, and `quat[3]` is asssumed to
        contain the scalar portion.
    degrees : bool, default=False
        If `degrees=False` (default), output Euler angles in radians. If True,
        convert these to degrees.
    normalize : bool, default=True
        If `normalize=True` (default), quaternions are scaled to have unit norm
        before converting to Euler angles.
    ignore_warnings : bool, default=True
        Set `ignore_warnings=True` (default) to suppress a `UserWarning` about
        gimbal lock, if it occurs.

    Returns
    -------
    angles : (3, n_angles) or (3,) array
        `quat` converted to Euler angle representation. `angles[0]` contains
        yaw, `angles[1]` contains pitch, and `angles[2]` contains roll.
    """
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore', category=UserWarning)
        angles = Rotation(np.asarray(quat).T, normalize=normalize)
        return angles.as_euler('ZYX', degrees=degrees).T


def euler_to_quaternion(angles, degrees=False):
    """
    Convert Euler angles to quaternion representation.

    Parameters
    ----------
    angles : (3, n_angles) or (3,) array
        Euler angles to convert to quaternion representation. `angles[0]`
        is assumed to contain yaw, `angles[1]` pitch, and `angles[2]` roll.
    degrees : bool, default=False
        If `degrees=False` (default), assumes `angles` are in radians. If True,
        assumes `angles` are in degrees.

    Returns
    -------
    quat : (4, n_angles) or (4,) array
        `angles` in quaternion representation. `quat[:3]` contains the vector
        portion of the quaternion, and `quat[3]` contains the scalar portion.
    """
    angles = Rotation.from_euler('ZYX', np.asarray(angles).T, degrees=degrees)
    return angles.as_quat().T
