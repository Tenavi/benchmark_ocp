import numpy as np


def dynamic_pressure(va, rhoS):
    """
    Evaluate dynamic pressure as a function of airspeed.

    Parameters
    ----------
    va : (n_points,) array
        Vehicle airspeed [m/s].
    rhoS : float
        `rhoS = 1/2 * rho * S`, i.e. one half air density times aerodynamic
        reference surface area.

    Returns
    -------
    pressure : (n_points,) array
        Dynamic pressure, `pressure = rhoS * va ** 2`
    """
    return rhoS * va ** 2


def blending_fun(alpha, alpha_stall, aero_blend_rate=50., jac=False):
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


def coefs_alpha(alpha, parameters, sin_alpha=None, cos_alpha=None, jac=False):
    """
    Compute contributions to the coefficients of lift (`CL`), drag (`CD`), and
    pitching moment (`CM`), from angle of attack (`alpha`). Uses the models in
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
            * `CM0` (float): intercept of pitching moment
            * `CMalpha` (float): pitching moment slope
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
    CL : (n_points,) array
        Lift coefficient from angle of attack.
    CD : (n_points,) array
        Drag coefficient from angle of attack.
    CM : (n_points,) array
        Pitching moment coefficient from angle of attack.
    d_CL : (n_points,) array
        Derivative of `CL` w.r.t. `alpha`. Only returned if jac=True.
    d_CD : (n_points,) array
        Derivative of `CD` w.r.t. `alpha`. Only returned if jac=True.
    d_CM : (n_points,) array
        Derivative of `CM` w.r.t. `alpha`. Only returned if jac=True.
    """

    # Linear components
    CL_lin = parameters.CL0 + parameters.CLalpha * alpha
    CD_lin = CL_lin / (np.pi * parameters.e * parameters.AR)
    if jac:
        d_CD_lin = (2. * parameters.CLalpha) * CD_lin
    CD_lin = parameters.CD0 + CL_lin * CD_lin
    CM = parameters.CM0 + parameters.CMalpha * alpha

    # Nonlinear adjustment for post-stall model
    sigma = blending_fun(alpha, parameters.alpha_stall,
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

    CL = sigma_inv * CL_lin + sigma * abs_2_sincos_alpha
    CD = sigma_inv * CD_lin + 2. * sigma * sin2_alpha

    if not jac:
        return CL, CD, CM

    d_CL = (parameters.CLalpha
            + sigma * (abs_2_sin_alpha * (2. * cos_alpha ** 2 - sin2_alpha)
                       - parameters.CLalpha)
            + d_sigma * (abs_2_sincos_alpha - CL_lin))

    d_CD = (sigma_inv * d_CD_lin
            + d_sigma * (2. * sin2_alpha - CD_lin)
            + sigma * 4. * sin_cos_alpha)

    d_CM = np.full_like(CM, parameters.CMalpha)

    return CL, CD, CM, d_CL, d_CD, d_CM


def aero_forces(states, controls):
    '''
    Compute the aerodynamic forces and moments based on airspeed, angle of
    attack, sideslip, angular rates, and control inputs.

    Parameters
    ----------
    states : VehicleState
        Current states.
    controls : Controls
        Control inputs.

    Returns
    -------
    forces : (3,) or (3, n_points) array
        Forces acting along body x, y, and z axes.
    moments : (3,) or (3, n_points) array
        Moments acting in body yaw, pitch, and roll directions.
    '''

    Va, alpha, beta = states.airspeed()

    forces = np.zeros((3, Va.shape[1]))
    moments = np.zeros((3, Va.shape[1]))

    # If airspeed is zero, no aerodynamics. Get index where airspeed is non-zero
    idx = Va > 0

    # Normalize angular rates
    p_bar = (parameters.b / 2.) * (states.p[idx] / Va[idx])
    q_bar = (parameters.c / 2.) * (states.q[idx] / Va[idx])
    r_bar = (parameters.b / 2.) * (states.r[idx] / Va[idx])

    ##### Longitudinal aerodynamics #####

    sin_alpha, cos_alpha = np.sin(alpha[idx]), np.cos(alpha[idx])

    # CL, CD, CM due to angle of attack
    CLalpha, CDalpha, CMalpha = coefs_alpha(alpha[idx], sin_alpha, cos_alpha)

    # CL, CD, CM due to pitch rate
    CLq, CDq, CMq = np.outer(
        [parameters.CLq, parameters.CDq, parameters.CMq], q_bar
    )

    # CL, CD, CM due to elevator deflection
    CLdeltaE, CDdeltaE, CMdeltaE = np.outer(
        [parameters.CLdeltaE, parameters.CDdeltaE, parameters.CMdeltaE],
        controls.elevator[idx]
    )

    # Pitching moment (m)
    moments[1,idx[0]] = parameters.c * (CMalpha + CMq + CMdeltaE)

    # Lift and drag, rotated into body frame (Fx, Fz)
    F_lift = CLalpha + CLq + CLdeltaE
    F_drag = CDalpha + CDq + CDdeltaE
    forces[0,idx[0]] = - cos_alpha * F_drag + sin_alpha * F_lift
    forces[2,idx[0]] = - sin_alpha * F_drag - cos_alpha * F_lift

    ##### Lateral aerodynamics #####

    # Sideslip contribution (linear models)
    FM_lat = np.reshape([parameters.CY0, parameters.Cl0, parameters.Cn0], (3,1))
    FM_lat = FM_lat + np.outer(
        [parameters.CYbeta, parameters.Clbeta, parameters.Cnbeta], beta[idx]
    )

    # Roll and yaw contributions
    FM_lat += np.outer(
        [parameters.CYp, parameters.Clp, parameters.Cnp], p_bar
    )
    FM_lat += np.outer(
        [parameters.CYr, parameters.Clr, parameters.Cnr], r_bar
    )

    # Control surface contributions
    FM_lat += np.outer(
        [parameters.CYdeltaA, parameters.CldeltaA, parameters.CndeltaA],
        controls.aileron[idx]
    )
    FM_lat += np.outer(
        [parameters.CYdeltaR, parameters.CldeltaR, parameters.CndeltaR],
        controls.rudder[idx]
    )

    forces[1,idx[0]] = FM_lat[0]
    moments[0,idx[0]] = parameters.b * FM_lat[1]
    moments[2,idx[0]] = parameters.b * FM_lat[2]

    pressure = dynamic_pressure(Va[idx])
    forces[:,idx[0]] *= pressure
    moments[:,idx[0]] *= pressure

    return np.squeeze(forces), np.squeeze(moments)


def prop_forces(Va, throttle):
    '''
    Simple propellor model, modified from Beard Chapter 4.3.

    Parameters
    ----------
    Va : (1, n_points) array
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
    forces = np.zeros((3, Va.shape[1]))
    moments = np.zeros((3, Va.shape[1]))

    coef = 0.5 * parameters.rho * parameters.Sprop * parameters.Cprop
    forces[0] = coef * (parameters.kmotor**2 * throttle - Va**2)
    moments[0] = -parameters.kTp * parameters.kOmega**2 * throttle

    return np.squeeze(forces), np.squeeze(moments)
