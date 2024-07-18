import numpy as np

from optimalcontrol.utilities import approx_derivative

from examples.common_utilities.dynamics import cross_product_matrix

from .containers import VehicleState, Controls


def dynamics(states, controls, parameters, aero_model):
    """
    Evaluate the state derivatives given states and controls.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.
    parameters : object
        Object containing mass and aerodynamic properties of the vehicle.
    aero_model : callable
        Function returning aero-propulsive forces and moments.

    Returns
    -------
    derivatives : VehicleState
        State dynamics, dx/dt.
    """
    forces, moments = aero_model(states, controls)
    return rigid_body_dynamics(states, forces, moments, parameters)


def jacobians(states, controls, parameters, aero_model,
              return_dfdx=True, return_dfdu=True):
    """
    Evaluate Jacobians of the state derivatives given states and controls.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.
    parameters : object
        Object containing mass and aerodynamic properties of the vehicle.
    aero_model : callable
        Function returning aero-propulsive forces and moments. It is assumed
        that forces and moments are functions of body frame velocities and rates
        only.
    return_dfdx : bool, default=True
        If `True`, compute the Jacobian with respect to states.
    return_dfdu : bool, default=True
        If `True`, compute the Jacobian with respect to controls.

    Returns
    -------
    dfdx : (11, 11) or (11, 11, n_points) array
        State Jacobians $df/dx (x,u)$ evaluated at `x=states` and `u=controls`.
    dfdu : (11, 4) or (11, 4, n_points) array
        Control Jacobians $df/du (x,u)$ evaluated at `x=states` and
        `u=controls`.
    """

    forces_0, moments_0 = aero_model(states, controls)
    aero_0 = np.concatenate([forces_0, moments_0], axis=0)

    # Jacobian with respect to states
    if return_dfdx:
        def aero_wrapper(v_w):
            x = states.to_array(copy=True)
            x[1:7] = v_w
            x = VehicleState.from_array(x)
            f_b, m_b = aero_model(x, controls)
            return np.concatenate([f_b, m_b], axis=0)

        daero_dx = approx_derivative(aero_wrapper, states.to_array()[1:7],
                                     f0=aero_0)
        dfdx = rigid_body_jac(states, daero_dx[:3], daero_dx[3:], parameters)
        if not return_dfdu:
            return dfdx

    # Jacobian with respect to controls
    if return_dfdu:
        def aero_wrapper(u):
            u_cont = Controls.from_array(u)
            f_b, m_b = aero_model(states, u_cont)
            return np.concatenate([f_b, m_b], axis=0)

        daero_du = approx_derivative(aero_wrapper, controls.to_array(),
                                     f0=aero_0)
        daero_du[:3] /= parameters.mass
        daero_du[3:] = np.einsum('ij,jk...->ik...', parameters.J_inv_body,
                                 daero_du[3:])
        dfdu = np.zeros((states.dim,) + daero_du.shape[1:])
        dfdu[1:7] = daero_du
        if not return_dfdx:
            return dfdu

    return dfdx, dfdu


def rigid_body_dynamics(states, forces, moments, parameters):
    """
    Evaluate the state dynamics given states, aero-propulsive forces and moments,
    and vehicle mass properties.

    Parameters
    ----------
    states : VehicleState
        Current states.
    forces : (3,) or (3, n_points) array
        Aero-propulsive forces acting in body frame along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Aero-propulsive moments acting in body roll, pitch, and yaw directions.
    parameters : object
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
    gravity = np.squeeze(states.rotation_matrix[2]) * parameters.g0
    d_vb = - np.cross(omega, vb, axis=0) + forces / parameters.mass + gravity

    # Angular rates (Beard (3.11))
    d_omega = np.matmul(parameters.J_body, omega)
    d_omega = - np.cross(omega, d_omega, axis=0) + moments
    d_omega = np.matmul(parameters.J_inv_body, d_omega)

    # Quaternions (Beard B.3)
    d_quat = np.empty_like(quat)
    d_quat[:-1] = 0.5 * (quat[-1:] * omega - np.cross(omega, quat[:-1], axis=0))
    d_quat[-1] = - 0.5 * np.einsum('i...,i...->...', omega, quat[:-1])

    return VehicleState(pd=d_pos[2], u=d_vb[0], v=d_vb[1], w=d_vb[2],
                        p=d_omega[0], q=d_omega[1], r=d_omega[2],
                        attitude=d_quat)


def rigid_body_jac(states, forces_jac, moments_jac, parameters):
    """
    Evaluate the Jacobians of the rigid body dynamics with respect to states.

    Parameters
    ----------
    states : VehicleState
        Current states.
    forces_jac : (3, 6) or (3, 6, n_points) array
        Jacobians of aero-propulsive forces with respect to body frame velocity
        (forces_jac[:, :3]) and rates (forces_jac[:, 3:]).
    moments_jac : (3, 6) or (3, 6, n_points) array
        Jacobians of aero-propulsive moments with respect to body frame velocity
        (moments_jac[:, :3]) and rates (moments_jac[:, 3:])
    parameters : object
        Object containing mass properties of the vehicle. Must have the
        following attributes:
            * `mass` (float): vehicle mass
            * `J_body` (3, 3) array: inertia matrix
            * `J_inv_body` (3, 3) array: inverse inertia matrix

    Returns
    -------
    dfdx : (11, 11) or (11, 11, n_points) array
        Jacobian of state dynamics with respect to states, df/dx.
    """
    vb = states.velocity
    omega = states.rates
    quat = states.attitude

    dfdx = np.zeros((states.dim,) + states.to_array().shape)

    forces_jac = np.reshape(forces_jac, (3, 6) + dfdx.shape[2:])
    moments_jac = np.reshape(moments_jac, forces_jac.shape)

    # Jacobian of last row of rotation matrix, for gravity and altitude
    q2 = 2. * quat
    d_R_d_quat = np.array([[q2[2], -q2[3], q2[0], -q2[1]],
                           [q2[3], q2[2], q2[1], q2[0]],
                           [-q2[0], -q2[1], q2[2], q2[3]]])

    wx = cross_product_matrix(omega)
    Jwx = cross_product_matrix(np.matmul(parameters.J_body, omega))
    qx = cross_product_matrix(quat[:-1])
    vx = cross_product_matrix(vb)

    diag_idx = np.diag_indices(3)
    qx[diag_idx[0], diag_idx[1]] += quat[-1]

    # Altitude dynamics
    #   w.r.t. velocity
    dfdx[0, 1:4] = np.squeeze(states.rotation_matrix[2])
    #   w.r.t. quaternion attitude
    dfdx[0, -4:] = np.einsum('ji...,j...->i...', d_R_d_quat, vb)

    # Velocity dynamics
    #   from forces
    dfdx[1:4, 1:7] = forces_jac / parameters.mass
    #   w.r.t. velocity
    dfdx[1:4, 1:4] -= wx
    #   w.r.t. rates
    dfdx[1:4, 4:7] += vx
    #   w.r.t. quaternions
    dfdx[1:4, 7:] = d_R_d_quat * parameters.g0

    # Rate dynamics
    #   from moments
    dfdx[4:7, 1:7] = moments_jac
    #   w.r.t. rates
    dfdx[4:7, 4:7] += Jwx - np.einsum('ij...,jk->ik...', wx, parameters.J_body)
    dfdx[4:7, 1:7] = np.einsum('ij,jk...->ik...', parameters.J_inv_body,
                               dfdx[4:7, 1:7])

    # Quaternion dynamics
    #   w.r.t. rates
    dfdx[7:10, 4:7] = qx
    dfdx[10, 4:7] = -quat[:-1]
    #   w.r.t. quaternions
    dfdx[7:10, 7:10] = -wx
    dfdx[7:10, 10] = omega
    dfdx[10, 7:10] = -omega

    dfdx[7:] *= 0.5

    return np.squeeze(dfdx)
