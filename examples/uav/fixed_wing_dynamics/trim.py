import numpy as np
from scipy import optimize

from examples.common_utilities.dynamics import euler_to_quaternion
from .dynamics import dynamics
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
    zero = np.zeros_like(pitch)
    angles = np.stack([zero, pitch, zero], axis=0)
    quaternion = euler_to_quaternion(angles)

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
    obj : (1,) array
        Discrepancy between the vector field evaluated at the current trim state
        and controls, and the desired vector field.
    """
    dxdt = dynamics(states, controls, parameters, aero_model)

    return np.sum(dxdt.to_array() ** 2, axis=0)


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
        return _trim_obj_fun(states, controls, parameters, aero_model)

    opt_res = optimize.minimize(fun=cost_fun_wrapper,
                                x0=xu_guess,
                                bounds=bounds,
                                **minimize_opts)

    trim_states, trim_controls = _split_opt_variable(opt_res.x, va_star)

    dxdt = dynamics(trim_states, trim_controls, parameters, aero_model)

    return trim_states, trim_controls, dxdt
