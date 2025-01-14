import numpy as np
from scipy import optimize

from .dynamics import dynamics
from .containers import VehicleState, Controls


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
    trim_states : `VehicleState`
    trim_controls : `Controls`
    dxdt
    """
    gamma_trim = 0.

    bounds = _make_bounds(parameters,
                          min_pitch=gamma_trim - np.deg2rad(30.),
                          max_pitch=gamma_trim + np.deg2rad(30.))

    xu_guess = (bounds.ub + bounds.lb) / 2.

    obj_fun, constraint_fun = _make_objective_and_constraint([1, 3, 4, 5, 6],
                                                             va_trim,
                                                             gamma_trim,
                                                             parameters,
                                                             aero_model)

    minimize_opts = {'method': 'trust-constr', 'jac': '3-point',
                     **minimize_opts}

    opt_res = optimize.minimize(fun=obj_fun, x0=xu_guess, bounds=bounds,
                                constraints=constraint_fun, **minimize_opts)

    trim_states, trim_controls = _split_opt_variable(opt_res.x,
                                                     va_trim,
                                                     gamma_trim)

    dxdt = dynamics(trim_states, trim_controls, parameters, aero_model)

    return trim_states, trim_controls, dxdt


def _split_opt_variable(xu, va_trim, gamma_trim=0.):
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


def _make_bounds(parameters,
                 min_pitch=np.deg2rad(-30.),
                 max_pitch=np.deg2rad(30.)):
    lb = np.concatenate((np.reshape(min_pitch, (1,)),
                         parameters.min_controls.to_array()))
    ub = np.concatenate((np.reshape(max_pitch, (1,)),
                         parameters.max_controls.to_array()))
    return optimize.Bounds(lb=lb, ub=ub)


def _make_objective_and_constraint(constr_idx, va_trim, gamma_trim, parameters,
                                   aero_model):
    # Indices of unconstrained state dynamics
    unconstr_idx = np.arange(VehicleState.dim)
    unconstr_idx = unconstr_idx[~np.isin(unconstr_idx, constr_idx)]

    def eval_dxdt(xu):
        states, controls = _split_opt_variable(xu, va_trim, gamma_trim)
        return dynamics(states, controls, parameters, aero_model)

    def obj_fun(xu):
        dxdt = eval_dxdt(xu)
        return np.sum(dxdt.to_array()[unconstr_idx] ** 2, axis=0)

    def constr_fun(xu):
        dxdt = eval_dxdt(xu)
        return dxdt.to_array()[constr_idx]

    constraint = optimize.NonlinearConstraint(constr_fun, lb=0., ub=0.)

    return obj_fun, constraint
