import numpy as np

from .containers import VehicleState


def dynamics(states, controls, parameters, aero_model):
    """
    Evaluate the state derivatives given states and controls.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.
    parameters : ProblemParameters
        Object containing mass and aerodynamic properties of the vehicle.
    aero_model : callable
        Function returning aero-propulsive forces and moments.

    Returns
    -------
    derivatives : VehicleState
        State dynamics, dx/dt.
    """
    forces, moments = aero_model(states, controls)
    forces += gravity(states, parameters.mg)

    return rigid_body_dynamics(states, forces, moments, parameters)


def rigid_body_dynamics(states, forces, moments, parameters):
    """
    Evaluate the state derivatives given states, forces, moments, and vehicle
    mass properties.

    Parameters
    ----------
    states : VehicleState
        Current states.
    forces : (3,) or (3, n_points) array
        Forces acting in body frame along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    parameters : ProblemParameters
        Object containing mass properties of the vehicle. Must have the
        following attributes:
            * `mass` (float): vehicle mass
            * `J_body` (3, 3) array: inertia matrix
            * `J_inv_body` (3, 3) array: inverse inertia matrix

    Returns
    -------
    derivatives : VehicleState
        State dynamics, dx/dt.
    """
    vb = states.velocity
    omega = states.rates
    quat = states.attitude

    forces = np.reshape(forces, vb.shape)
    moments = np.reshape(moments, omega.shape)

    # Inertial position (Beard (B.1))
    d_pos = states.body_to_inertial(vb)

    # Inertial velocity (Beard (3.7))
    d_vb = - np.cross(omega, vb, axis=0) + forces / parameters.mass

    # Quaternions (Beard B.3)
    d_quat = np.empty_like(quat)
    d_quat[:-1] = 0.5 * (quat[-1:] * omega - np.cross(omega, quat[:-1], axis=0))
    d_quat[-1] = - 0.5 * np.einsum('i...,i...->...', omega, quat[:-1])

    # Angular rates (Beard (3.11))
    d_omega = np.matmul(parameters.J_body, omega)
    d_omega = - np.cross(omega, d_omega, axis=0) + moments
    d_omega = np.matmul(parameters.J_inv_body, d_omega)

    return VehicleState(pd=d_pos[2], u=d_vb[0], v=d_vb[1], w=d_vb[2],
                        p=d_omega[0], q=d_omega[1], r=d_omega[2],
                        attitude=d_quat)


def gravity(states, mg):
    """
    Compute the force of gravity acting in the body frame.

    Parameters
    ----------
    states : VehicleState
        Current states.
    mg : float
        `mg = mass * g0`, i.e. vehicle mass times constant of gravity.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Gravity expressed in body frame.
    """
    return states.inertial_to_body([0., 0., mg])
