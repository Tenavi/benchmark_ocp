import numpy as np
from scipy import optimize

from .dynamics import dynamics, jacobians
from .containers import VehicleState, Controls


_n_free = 1 + Controls.dim


def _split_opt_variable(xu, va_star):
    """

    Parameters
    ----------
    opt_variable
    va_star

    Returns
    -------

    """
    controls = Controls.from_array(xu[1:])

    pitch = xu[0]
    half_pitch = pitch / 2.
    quaternion = np.zeros((4,) + pitch.shape[1:])
    quaternion[1] = np.sin(half_pitch)
    quaternion[3] = np.cos(half_pitch)

    # Velocity in body x and z directions
    u = va_star * np.cos(pitch)
    w = va_star * np.sin(pitch)

    states = VehicleState(u=u, w=w, attitude=quaternion)

    return states, controls


def _make_bounds(parameters):
    """

    Parameters
    ----------
    parameters

    Returns
    -------

    """
    lb = np.empty((_n_free,))
    ub = np.empty((_n_free,))

    # Pitch [rad]
    lb[0] = -np.pi / 4.
    ub[0] = np.pi / 4.

    # Control constraints
    lb[1:] = parameters.min_controls.to_array()
    ub[1:] = parameters.max_controls.to_array()

    return optimize.Bounds(lb=lb, ub=ub)


def _trim_obj_fun(states, controls, parameters, aero_model):
    """
    Parameters
    ----------
    states : VehicleState
        Trim state.
    controls : Controls
        Trim controls.
    parameters : object
    aero_model : callable

    Returns
    -------
    dxdt_norm : (1,) array
        Discrepancy between the vector field evaluated at the current trim state
        and controls, and the desired vector field.
    grad : (
    """
    dxdt = dynamics(states, controls, parameters, aero_model)
    dfdx, dfdu = jacobians(states, controls, parameters, aero_model)

    dxdt_norm = 0.5 * np.sum(dxdt.to_array() ** 2, axis=0)

    grad_x = np.einsum('i...,ij...->j...', dxdt.to_array(), dfdx)
    grad_u = np.einsum('i...,ij...->j...', dxdt.to_array(), dfdu)

    return dxdt_norm, grad_x, grad_u


def compute_trim(va_star, parameters, aero_model, **minimize_opts):
    """
    Compute the trim state given a desired airspeed, constant turn radius, and
    constant flight path angle. Uses constrained optimization.

    Parameters
    ----------
    va_star : float
        Desired trim airspeed [m/s].
    parameters

    Returns
    -------
    trim_states : VehicleState
        Trim state. pn, pe, pd are not set.
    trim_controls : Controls
        Trim controls.
    dxdt
    """
    bounds = _make_bounds(parameters)

    xu_guess = (bounds.ub + bounds.lb) / 2.

    def cost_fun_wrapper(xu):
        states, controls = _split_opt_variable(xu, va_star)
        dxdt_norm, grad_x, grad_u = _trim_obj_fun(states, controls,
                                                  parameters, aero_model)

        # Chain rule for derivatives of state to derivatives of pitch
        d_u_d_pitch = -states.w
        d_w_d_pitch = states.u
        d_q1_d_pitch = -0.5 * states.attitude[3]
        d_q3_d_pitch = 0.5 * states.attitude[1]

        grad_pitch = (grad_x[1] * d_u_d_pitch + grad_x[3] * d_w_d_pitch
                      + grad_x[8] * d_q1_d_pitch + grad_x[10] * d_q3_d_pitch)
        grad = np.concatenate([grad_pitch, grad_u], axis=0)

        return dxdt_norm, grad

    opt_res = optimize.minimize(fun=cost_fun_wrapper,
                                jac=True,
                                x0=xu_guess,
                                bounds=bounds,
                                **minimize_opts)

    trim_states, trim_controls = _split_opt_variable(opt_res.x, va_star)

    dxdt = dynamics(trim_states, trim_controls, parameters, aero_model)

    return trim_states, trim_controls, dxdt
