"""
##### References
1. R. W. Beard and T. W. McClain, Small Unmanned Aircraft: Theory and Practice,
    Princeton University Press, Princeton, NJ, 2012.
2. R. W. Beard and T. W. McClain, Small Unmanned Aircraft: Theory and Practice
    [Supplement], Princeton University Press, Princeton, NJ, 2nd. ed., 2022.
"""

import numpy as np

from . import constants


def aeroprop_forces(states, controls, params=constants):
    """
    Compute the aero-propulsive (aerodynamic and propulsive) forces and moments
    based on airspeed, angle of attack, sideslip, angular rates, and control
    inputs.

    Parameters
    ----------
    states : `VehicleState`
        Current states.
    controls : `Controls`
        Control inputs.
    params : object

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body roll, pitch, and yaw directions.
    """
    va, _, _ = states.airspeed

    forces, moments = aero_forces(states, controls, params=params)
    prop_thrust, prop_torque = prop_forces(va, controls.throttle, params=params)

    forces[:1] += prop_thrust
    moments[:1] -= prop_torque

    return forces, moments


def aero_forces(states, controls, params=constants):
    """
    Compute the aerodynamic forces and moments based on airspeed, angle of
    attack, sideslip, angular rates, and control inputs.

    Parameters
    ----------
    states : `VehicleState`
        Current states.
    controls : `Controls`
        Control inputs.
    params : object

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body roll, pitch, and yaw directions.
    """
    va, alpha, beta = states.airspeed

    forces = np.zeros((3, states.n_points))
    moments = np.zeros((3, states.n_points))

    # If airspeed is zero, no aerodynamics. Get index where airspeed is non-zero
    idx = va > np.finfo(float).eps

    if idx.any():
        # Fx, Fz, My
        forces[0, idx], forces[2, idx], moments[1, idx] = _longitudinal_aero(
            alpha[idx], va[idx], states.q[idx], controls.elevator[idx])

        # Fy, Mx, Mz
        forces[1, idx], moments[0, idx], moments[2, idx] = _lateral_aero(
            beta[idx], va[idx], states.p[idx], states.r[idx],
            controls.aileron[idx], controls.rudder[idx])

        # Multiply by dynamic pressure * S
        pressure = params.half_rhoS * va[idx] ** 2
        forces[:, idx] *= pressure
        moments[:, idx] *= pressure

    return np.squeeze(forces), np.squeeze(moments)


def _longitudinal_aero(alpha, va, q, elevator, params=constants):
    """
    Evaluate longitudinal forces and moments, without multiplying by dynamic
    pressure (this is assumed to be multiplied outside of this function).

    Parameters
    ----------
    alpha
    va
    q
    elevator
    params

    Returns
    -------

    """
    # Normalize pitch rate
    q = (params.c / 2.) * q / va

    # CL, CD, Cm due to angle of attack
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)

    coefs = _coefs_alpha(alpha, sin_alpha=sin_alpha, cos_alpha=cos_alpha)

    # CL, CD, Cm due to pitch rate and elevator deflection
    coefs += np.einsum('i,...->i...',
                       [params.CLq, params.CDq, params.Cmq],
                       q).reshape(coefs.shape)
    coefs += np.einsum('i,...->i...',
                       [params.CLdeltaE, params.CDdeltaE, params.CmdeltaE],
                       elevator).reshape(coefs.shape)

    # Pitching moment times chord length
    coefs[2] *= params.c

    # Lift and drag, rotated into body frame
    rotation = np.array([[-sin_alpha, cos_alpha],
                         [cos_alpha, sin_alpha]])

    coefs[:2] = np.einsum('ij...,j...->i...', rotation, -coefs[:2])

    return coefs


def _lateral_aero(beta, va, p, r, aileron, rudder, params=constants):
    """
    Evaluate lateral forces and moments, without multiplying by dynamic pressure
    (this is assumed to be multiplied outside of this function).

    Parameters
    ----------
    beta
    va
    p
    r
    aileron
    rudder
    params

    Returns
    -------

    """
    # Normalize angular rates
    b_2va = (params.b / 2.) / va
    p = p * b_2va
    r = r * b_2va

    # Sideslip contributions
    coefs = np.einsum('i,...->i...',
                      [params.CYbeta, params.Clbeta, params.Cnbeta],
                      beta)

    # Roll and yaw rate contributions
    coefs += np.einsum('i,...->i...',
                       [params.CYp, params.Clp, params.Cnp],
                       p).reshape(coefs.shape)
    coefs += np.einsum('i,...->i...',
                       [params.CYr, params.Clr, params.Cnr],
                       r).reshape(coefs.shape)

    # Control surface contributions
    coefs += np.einsum('i,...->i...',
                       [params.CYdeltaA, params.CldeltaA, params.CndeltaA],
                       aileron).reshape(coefs.shape)
    coefs += np.einsum('i,...->i...',
                       [params.CYdeltaR, params.CldeltaR, params.CndeltaR],
                       rudder).reshape(coefs.shape)

    # Constant offsets
    const_coefs = [params.CY0, params.Cl0, params.Cn0]
    if coefs.ndim == 1:
        coefs += const_coefs
    else:
        coefs += np.reshape(const_coefs, (3, 1))

    # Moments
    coefs[1:] *= params.b

    return coefs


def _coefs_alpha(alpha, sin_alpha=None, cos_alpha=None, jac=False,
                 params=constants):
    """
    Compute contributions to the coefficients of lift (`CL`), drag (`CD`), and
    pitching moment (`Cm`), from angle of attack (`alpha`). Uses the models in
    Beard Chapter 4, with a modified model for post-stall drag. Optionally also
    compute the derivatives of each coefficient with respect to `alpha`.

    Parameters
    ----------
    alpha : (n_points,) array
        Angle of attack [rad].
    sin_alpha : (n_points,) array, optional
        Pre-computed `sin(alpha)`, if available.
    cos_alpha : (n_points,) array, optional
        Pre-computed `cos(alpha)`, if available.
    jac : bool, default=False
        If `jac=True`, also compute the derivatives with respect to `alpha`.
    params : object

    Returns
    -------
    coefs : (3, n_points) array
        Lift, drag, and pitching moment coefficient contributions from angle of
        attack.
    jacs : (3, n_points) array
        Derivatives of each aero coefficient with respect to `alpha`. Only
        returned if jac=True.
    """
    alpha = np.asarray(alpha)
    coefs = np.empty((3,) + alpha.shape)

    # Linear components
    CL_lin = params.CL0 + params.CLalpha * alpha
    CD_lin = CL_lin / (np.pi * params.eos * params.AR)
    if jac:
        d_CD_lin = (2. * params.CLalpha) * CD_lin
    CD_lin = params.CD0 + CL_lin * CD_lin
    coefs[2] = params.Cm0 + params.Cmalpha * alpha

    # Nonlinear adjustment for post-stall model
    sigma = _blending_fun(alpha, params.alpha_stall,
                          aero_blend_rate=params.aero_blend_rate, jac=jac)
    if jac:
        sigma, d_sigma = sigma
    sigma_inv = 1. - sigma

    if sin_alpha is None:
        sin_alpha = np.sin(alpha)
    if cos_alpha is None:
        cos_alpha = np.cos(alpha)

    sin2_alpha = sin_alpha ** 2
    sin_cos_alpha = sin_alpha * cos_alpha
    abs_2_sin_alpha = 2. * np.abs(sin_alpha)
    abs_2_sincos_alpha = abs_2_sin_alpha * sin_cos_alpha

    coefs[0] = sigma_inv * CL_lin + sigma * abs_2_sincos_alpha
    coefs[1] = sigma_inv * CD_lin + 2. * sigma * sin2_alpha

    if not jac:
        return coefs

    jacs = np.empty_like(coefs)

    jacs[0] = (params.CLalpha
               + sigma * (abs_2_sin_alpha * (2. * cos_alpha ** 2 - sin2_alpha)
                          - params.CLalpha)
               + d_sigma * (abs_2_sincos_alpha - CL_lin))

    jacs[1] = (sigma_inv * d_CD_lin
               + d_sigma * (2. * sin2_alpha - CD_lin)
               + sigma * 4. * sin_cos_alpha)

    jacs[2] = params.Cmalpha

    return coefs, jacs


def _blending_fun(alpha, alpha_stall, aero_blend_rate=50., jac=False):
    """
    Evaluates the sigmoid-type blending function from Beard (4.10), and,
    optionally, its derivative.

    Parameters
    ----------
    alpha : (n_points,) or (n_points) array
        Angle of attack [rad].
    alpha_stall : float
        Stall angle of attack [rad].
    aero_blend_rate : float, default=50
        Parameter determining the steepness of the transition between 0 and 1.
    jac : bool, default=False
        If `jac=True`, also compute the derivative with respect to `alpha`.

    Returns
    -------
    sigma : (n_points,) array
        Smooth blending function which is approximately 0 for
        `-alpha_stall < alpha < alpha_stall` and approximately 1 outside of this
        region.
    d_sigma : (n_points,) array
        Derivative d_sigma / d_alpha. Only returned if jac=True.
    """
    e_alpha = np.exp(aero_blend_rate * alpha)
    e_alpha0 = np.exp(aero_blend_rate * alpha_stall)
    e_alpha0_2 = e_alpha0 ** 2

    e_p = e_alpha0 * e_alpha
    e_m = e_alpha0 / e_alpha

    numerator = 1. + e_m + e_p
    # Equivalent to (1. + e_m) * (1. + e_p)
    denominator = numerator + e_alpha0_2
    sigma = numerator / denominator

    if not jac:
        return sigma

    d_sigma = aero_blend_rate * e_alpha0_2 * (e_p - e_m) / denominator ** 2

    return sigma, d_sigma


def prop_forces(va, throttle, params=constants):
    """
    Propeller model from Beard supplement Chapter 4.3.

    Parameters
    ----------
    va : (n_points) array
        Airspeed for each state [m/s].
    throttle : (n_points) array
        Throttle setting corresponding to each state.
    params : object

    Returns
    -------
    thrust : (n_points,) array
        Force acting along body x-axis.
    torque : (n_points,) array
        Moment acting in negative roll direction.
    """
    va = np.asarray(va)
    va_2 = va ** 2

    # Throttle to voltage
    voltage = np.asarray(throttle) * params.V_max

    # Compute propeller speed
    a = params.C_Q0 * params.rho_D_5 / (4. * np.pi ** 2)
    b = ((params.C_Q1 * params.rho_D_4 / (2. * np.pi)) * va
         + params.KQ * params.KV / params.R_motor)
    c = ((params.C_Q2 * params.rho_D_3) * va_2
         - (params.KQ / params.R_motor) * voltage
         + params.KQ * params.i0)

    # Propeller speed in [rad/s]
    omega = (- b + np.sqrt(np.maximum(b ** 2 - 4. * a * c, 0.))) / (2. * a)

    # Convert to [rot/s]
    omega = omega / (2. * np.pi)

    # Instead of computing advance ratio and dimensionless thrust and torque
    # coefficients, multiply airspeed (va) and propeller diameter (D_prop)
    # through thrust and torque equations
    omega_2 = omega ** 2
    va_omega = va * omega

    thrust = ((params.rho_D_4 * params.C_T0) * omega_2
              + (params.rho_D_3 * params.C_T1) * va_omega
              + (params.rho_D_2 * params.C_T2) * va_2)

    torque = ((params.rho_D_5 * params.C_Q0) * omega_2
              + (params.rho_D_4 * params.C_Q1) * va_omega
              + (params.rho_D_3 * params.C_Q2) * va_2)

    return thrust, torque
