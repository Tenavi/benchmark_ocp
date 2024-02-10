import numpy as np


def aero_forces(states, controls, parameters):
    """
    Compute the aerodynamic forces and moments based on airspeed, angle of
    attack, sideslip, angular rates, and control inputs.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.
    parameters : ProblemParameters

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    """

    va, alpha, beta = states.airspeed

    forces = np.zeros((3, states.n_points))
    moments = np.zeros((3, states.n_points))

    # If airspeed is zero, no aerodynamics. Get index where airspeed is non-zero
    idx = va > np.finfo(float).eps

    if idx.any():
        # Fx, Fz, My
        forces[0, idx], forces[2, idx], moments[1, idx] = _longitudinal_aero(
            alpha[idx], va[idx], states.q[idx], controls.elevator[idx],
            parameters)

        # Fy, Mx, Mz
        forces[1, idx], moments[0, idx], moments[2, idx] = _lateral_aero(
            beta[idx], va[idx], states.p[idx], states.r[idx],
            controls.aileron[idx], controls.rudder[idx], parameters)

        # Multiply by dynamic pressure * S
        pressure = parameters.rhoS * va[idx] ** 2
        forces[:, idx] *= pressure
        moments[:, idx] *= pressure

    return np.squeeze(forces), np.squeeze(moments)


def _longitudinal_aero(alpha, va, q, elevator, parameters):
    """
    Evaluate the longitudinal forces and moments, without multiplying by
    dynamic pressure (this is assumed to be multiplied outside of this
    function).

    Parameters
    ----------
    alpha
    va
    q
    elevator
    parameters

    Returns
    -------

    """

    # Normalize pitch rate
    q = (parameters.c / 2.) * q / va

    # CL, CD, Cm due to angle of attack
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)

    coefs = _coefs_alpha(alpha, parameters,
                         sin_alpha=sin_alpha, cos_alpha=cos_alpha)

    # CL, CD, Cm due to pitch rate and elevator deflection
    coefs += np.outer([parameters.CLq, parameters.CDq, parameters.Cmq], q)
    coefs += np.outer(
        [parameters.CLdeltaE, parameters.CDdeltaE, parameters.CmdeltaE],
        elevator)

    # Pitching moment times chord length
    coefs[2] *= parameters.c

    # Lift and drag, rotated into body frame
    coefs[:2] = [sin_alpha * coefs[0] - cos_alpha * coefs[1],
                 - cos_alpha * coefs[0] - sin_alpha * coefs[1]]

    return coefs


def _lateral_aero(beta, va, p, r, aileron, rudder, parameters):
    """
    Evaluate the lateral forces and moments, without multiplying by dynamic
    pressure (this is assumed to be multiplied outside of this function).

    Parameters
    ----------
    beta
    va
    p
    r
    aileron
    rudder
    parameters

    Returns
    -------

    """

    # Normalize angular rates
    p = (parameters.b / 2.) * p / va
    r = (parameters.b / 2.) * r / va

    # Sideslip contributions
    coefs = np.outer([parameters.CYbeta, parameters.Clbeta, parameters.Cnbeta],
                     beta)

    # Roll and yaw rate contributions
    coefs += np.outer([parameters.CYp, parameters.Clp, parameters.Cnp], p)
    coefs += np.outer([parameters.CYr, parameters.Clr, parameters.Cnr], r)

    # Control surface contributions
    coefs += np.outer(
        [parameters.CYdeltaA, parameters.CldeltaA, parameters.CndeltaA],
        aileron)
    coefs += np.outer(
        [parameters.CYdeltaR, parameters.CldeltaR, parameters.CndeltaR],
        rudder)

    coefs += np.reshape([parameters.CY0, parameters.Cl0, parameters.Cn0],
                        (3, 1))

    # Moments
    coefs[1:] *= parameters.b

    return coefs


def _coefs_alpha(alpha, parameters, sin_alpha=None, cos_alpha=None, jac=False):
    """
    Compute contributions to the coefficients of lift (`CL`), drag (`CD`), and
    pitching moment (`Cm`), from angle of attack (`alpha`). Uses the models in
    Beard Chapter 4, with a modified model for post-stall drag. Optionally also
    compute the derivatives of each coefficient with respect to `alpha`.

    Parameters
    ----------
    alpha : (n_points,) array
        Angle of attack [rad].
    parameters : ProblemParameters
        Object containing aerodynamic coefficients of the vehicle. Must have the
        following attributes:
            * `CL0` (float): intercept of lift coefficient
            * `CLalpha` (float): lift coefficient slope
            * `CD0` (float): parasitic drag
            * `Cm0` (float): intercept of pitching moment
            * `Cmalpha` (float): pitching moment slope
            * `e` (float): Oswald's efficiency factor
            * `AR` (float): aspect ratio
            * `alpha_stall` (float): stall angle of attack
            * `aero_blend_rate` (float): parameter in stall blending function
    sin_alpha : (n_points,) array, optional
        Pre-computed `sin(alpha)`, if available.
    cos_alpha : (n_points,) array, optional
        Pre-computed `cos(alpha)`, if available.
    jac : bool, default=False
        If `jac=True`, also compute the derivatives with respect to `alpha`.

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
    CL_lin = parameters.CL0 + parameters.CLalpha * alpha
    CD_lin = CL_lin / (np.pi * parameters.eos * parameters.AR)
    if jac:
        d_CD_lin = (2. * parameters.CLalpha) * CD_lin
    CD_lin = parameters.CD0 + CL_lin * CD_lin
    coefs[2] = parameters.Cm0 + parameters.Cmalpha * alpha

    # Nonlinear adjustment for post-stall model
    sigma = _blending_fun(alpha, parameters.alpha_stall,
                          aero_blend_rate=parameters.aero_blend_rate, jac=jac)
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

    jacs[0] = (parameters.CLalpha
               + sigma * (abs_2_sin_alpha * (2. * cos_alpha ** 2 - sin2_alpha)
                          - parameters.CLalpha)
               + d_sigma * (abs_2_sincos_alpha - CL_lin))

    jacs[1] = (sigma_inv * d_CD_lin
               + d_sigma * (2. * sin2_alpha - CD_lin)
               + sigma * 4. * sin_cos_alpha)

    jacs[2] = parameters.Cmalpha

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


def prop_forces(va, throttle):
    '''
    Simple propellor model, modified from Beard Chapter 4.3.

    Parameters
    ----------
    va : (1, n_points) array
        Airspeed for each state [m/s].
    throttle : (1, n_points) array
        Throttle setting corresponding to each state.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    '''
    forces = np.zeros((3, va.shape[1]))
    moments = np.zeros((3, va.shape[1]))

    coef = 0.5 * parameters.rho * parameters.Sprop * parameters.Cprop
    forces[0] = coef * (parameters.kmotor**2 * throttle - va**2)
    moments[0] = -parameters.kTp * parameters.kOmega**2 * throttle

    return np.squeeze(forces), np.squeeze(moments)
