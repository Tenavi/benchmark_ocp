import numpy as np
from scipy import optimize

from .dynamics import dynamics
from .containers import VehicleState, Controls


_n_free_vars = 1 + Controls.dim


def _split_opt_variable(xu, va_trim, gamma_trim=0.):
    """

    Parameters
    ----------
    opt_variable
    va_trim

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
    alpha = pitch - gamma_trim
    u = va_trim * np.cos(alpha)
    w = va_trim * np.sin(alpha)

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
    lb = np.empty((_n_free_vars,))
    ub = np.empty((_n_free_vars,))

    # Pitch [rad]
    lb[0] = -np.pi / 6.
    ub[0] = np.pi / 6.

    # Control constraints
    lb[1:] = parameters.min_controls.to_array()
    ub[1:] = parameters.max_controls.to_array()

    return optimize.Bounds(lb=lb, ub=ub)


def _make_objective_and_constraint(constr_idx, va_trim, parameters, aero_model):
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
    min_idx = np.arange(VehicleState.dim)
    min_idx = min_idx[~np.isin(min_idx, constr_idx)]

    def obj_fun(xu):
        states, controls = _split_opt_variable(xu, va_trim)
        dxdt = dynamics(states, controls, parameters, aero_model)
        return np.sum(dxdt.to_array()[min_idx] ** 2, axis=0)

    def constr_fun(xu):
        states, controls = _split_opt_variable(xu, va_trim)
        dxdt = dynamics(states, controls, parameters, aero_model)
        return dxdt.to_array()[constr_idx]

    constraint = optimize.NonlinearConstraint(constr_fun, lb=0., ub=0.)

    return obj_fun, constraint


def compute_trim(va_trim, parameters, aero_model, **minimize_opts):
    """
    Compute the trim state given a desired airspeed, constant turn radius, and
    constant flight path angle. Uses constrained optimization.

    Parameters
    ----------
    va_trim : float
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

    obj_fun, constraint_fun = _make_objective_and_constraint([1, 3, 4, 5, 6],
                                                             va_trim,
                                                             parameters,
                                                             aero_model)

    minimize_opts = {'method': 'trust-constr', 'jac': '3-point',
                     **minimize_opts}

    opt_res = optimize.minimize(fun=obj_fun, x0=xu_guess, bounds=bounds,
                                constraints=constraint_fun, **minimize_opts)

    trim_states, trim_controls = _split_opt_variable(opt_res.x, va_trim)

    dxdt = dynamics(trim_states, trim_controls, parameters, aero_model)

    return trim_states, trim_controls, dxdt
