import numpy as np
import pytest

from optimalcontrol.utilities import approx_derivative

from examples.uav.dynamics_model import aero
from examples.uav.dynamics_model.parameters import aerosonde as constants
from examples.uav.dynamics_model.tests.test_containers import (random_states,
                                                               random_controls)


rng = np.random.default_rng()


@pytest.mark.parametrize('n_points', [1, 2])
def test_blending_fun(n_points):
    a0, m = constants.alpha_stall, constants.aero_blend_rate

    alpha = rng.normal(scale=a0, size=(n_points,))

    sigma = aero._blending_fun(alpha, a0)

    num = 1. + np.exp(-m * (alpha - a0)) + np.exp(m * (alpha + a0))
    den = (1. + np.exp(-m * (alpha - a0))) * (1. + np.exp(m * (alpha + a0)))
    sigma_expect = num / den

    np.testing.assert_allclose(sigma, sigma_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_blending_fun_jac(n_points):
    a0, m = constants.alpha_stall, constants.aero_blend_rate

    alpha = rng.normal(scale=a0, size=(n_points,))

    _, sigma_d = aero._blending_fun(alpha, a0, aero_blend_rate=m, jac=True)

    sigma_d_expect = approx_derivative(lambda a: aero._blending_fun(a, a0),
                                       alpha.reshape(1, -1))
    sigma_d_expect = np.squeeze(sigma_d_expect)

    np.testing.assert_allclose(sigma_d, sigma_d_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_small_alpha(n_points):
    """Test that the aero coefficients are approximately linear (quadratic in
    the case of CD) close to alpha = 0."""
    alpha = rng.normal(scale=1e-07, size=(n_points,))

    CL, CD, Cm = aero._coefs_alpha(alpha, constants)

    CL_expect = constants.CL0 + constants.CLalpha * alpha

    CD_expect = np.pi * constants.eos * constants.b ** 2 / constants.S
    CD_expect = constants.CD0 + CL_expect ** 2 / CD_expect

    Cm_expect = constants.Cm0 + constants.Cmalpha * alpha

    np.testing.assert_allclose(CL, CL_expect, atol=1e-07)
    np.testing.assert_allclose(CD, CD_expect, atol=1e-07)
    np.testing.assert_allclose(Cm, Cm_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_high_alpha(n_points):
    """Test that the aero coefficients are approximately equal to the flat plate
    model past stall."""
    a0 = constants.alpha_stall
    alpha = rng.uniform(low=2. * a0, high=np.pi, size=(n_points,))

    CL, CD, Cm = aero._coefs_alpha(alpha, constants)

    CL_expect = 2. * np.sign(alpha) * (np.sin(alpha) ** 2 * np.cos(alpha))

    CD_expect = 2. * np.sin(alpha) ** 2

    Cm_expect = constants.Cm0 + constants.Cmalpha * alpha

    np.testing.assert_allclose(CL, CL_expect, atol=1e-07)
    np.testing.assert_allclose(CD, CD_expect, atol=1e-07)
    np.testing.assert_allclose(Cm, Cm_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_jac(n_points):
    alpha = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))

    _, jac = aero._coefs_alpha(alpha, constants, jac=True)
    d_CL, d_CD, d_Cm = jac

    assert d_CL.shape == d_CD.shape == d_Cm.shape == alpha.shape

    for i in range(n_points):
        d_coef_expect = np.squeeze(approx_derivative(
            lambda a: aero._coefs_alpha(a, constants), alpha[i]))
        np.testing.assert_allclose(d_CL[i], d_coef_expect[0], atol=1e-07)
        np.testing.assert_allclose(d_CD[i], d_coef_expect[1], atol=1e-07)
        np.testing.assert_allclose(d_Cm[i], d_coef_expect[2], atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_longitudinal_aero_zero_alpha(n_points):
    alpha = np.zeros(n_points)
    va = rng.uniform(low=1., high=10., size=n_points)
    q = rng.normal(size=n_points)
    elevator = rng.normal(size=n_points)

    CX, CZ, Cm = aero._longitudinal_aero(alpha, va, q, elevator, constants)

    q_bar = q * constants.c / (2. * va)

    # When alpha = 0, expect CL = -CZ
    CZ_expect = -(constants.CL0 + constants.CLq * q_bar
                  + constants.CLdeltaE * elevator)

    # When alpha = 0, expect CD = -CX
    denominator = np.pi * constants.eos * constants.b ** 2 / constants.S
    CX_expect = -(constants.CD0 + constants.CL0 ** 2 / denominator
                  + constants.CDq * q_bar + constants.CDdeltaE * elevator)

    Cm_expect = constants.c * (constants.Cm0 + constants.Cmq * q_bar
                               + constants.CmdeltaE * elevator)

    np.testing.assert_allclose(CX, CX_expect, atol=1e-07)
    np.testing.assert_allclose(CZ, CZ_expect, atol=1e-07)
    np.testing.assert_allclose(Cm, Cm_expect, atol=1e-07)


def test_longitudinal_aero_high_alpha():
    alpha = np.array([np.pi / 2., -np.pi / 2.])
    va = rng.uniform(low=1., high=10., size=2)
    q = rng.normal(size=2)
    elevator = rng.normal(size=2)

    CX, CZ, Cm = aero._longitudinal_aero(alpha, va, q, elevator, constants)

    q_bar = q * constants.c / (2. * va)

    CL_expect, CD_expect, Cm_expect = aero._coefs_alpha(alpha, constants)

    CL_expect += constants.CLq * q_bar + constants.CLdeltaE * elevator
    CD_expect += constants.CDq * q_bar + constants.CDdeltaE * elevator
    Cm_expect += constants.Cmq * q_bar + constants.CmdeltaE * elevator
    Cm_expect *= constants.c

    np.testing.assert_allclose(Cm, Cm_expect, atol=1e-07)
    # When alpha = 90 degrees, lift vector points +x, drag in -z
    np.testing.assert_allclose(CX[0], CL_expect[0], atol=1e-07)
    np.testing.assert_allclose(CZ[0], -CD_expect[0], atol=1e-07)
    # When alpha = -90 degrees, lift vector points -x, drag in +z
    np.testing.assert_allclose(CX[1], -CL_expect[1], atol=1e-07)
    np.testing.assert_allclose(CZ[1], CD_expect[1], atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
@pytest.mark.parametrize('non_zero_arg', ['beta', 'p', 'r', 'deltaA', 'deltaR'])
def test_lateral_aero(n_points, non_zero_arg):
    # Pick an airspeed that doesn't change the normalization on p and r
    args = dict(beta=np.zeros(n_points),
                va=np.full((n_points,), constants.b / 2.),
                p=np.zeros(n_points),
                r=np.zeros(n_points),
                deltaA=np.zeros(n_points),
                deltaR=np.zeros(n_points))

    args[non_zero_arg] = rng.normal(size=n_points)

    CY, Cl, Cn = aero._lateral_aero(*args.values(), constants)

    expected = {}
    for coef in ['CY', 'Cl', 'Cn']:
        a = getattr(constants, coef + '0')
        b = getattr(constants, coef + non_zero_arg)
        expected[coef] = a + b * args[non_zero_arg]
    np.testing.assert_allclose(CY, expected['CY'], atol=1e-14)
    np.testing.assert_allclose(Cl, constants.b * expected['Cl'], atol=1e-14)
    np.testing.assert_allclose(Cn, constants.b * expected['Cn'], atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
@pytest.mark.parametrize('zero_idx', [0, -1, []])
def test_aero_forces(n_points, zero_idx):
    states = random_states(n_points)
    controls = random_controls(n_points)

    states.u[zero_idx] = 0.
    states.v[zero_idx] = 0.
    states.w[zero_idx] = 0.

    va, alpha, beta = states.airspeed

    np.testing.assert_equal(va.reshape(-1)[zero_idx], 0.)

    forces, moments = aero.aero_forces(states, controls, constants)

    # Before reshaping, confirm that the correct shape was produced
    assert forces.shape == moments.shape == states.velocity.shape

    forces = forces.reshape(3, n_points)
    moments = moments.reshape(3, n_points)

    np.testing.assert_equal(forces[:, zero_idx], 0.)
    np.testing.assert_equal(moments[:, zero_idx], 0.)

    non_zero_idx = va > np.finfo(float).eps

    if non_zero_idx.any():
        pressure = 0.5 * constants.rho * constants.S * va[non_zero_idx] ** 2

        Cx, Cz, My = aero._longitudinal_aero(alpha[non_zero_idx],
                                             va[non_zero_idx],
                                             states.q[non_zero_idx],
                                             controls.elevator[non_zero_idx],
                                             constants)

        Cy, Mx, Mz = aero._lateral_aero(beta[non_zero_idx],
                                        va[non_zero_idx],
                                        states.p[non_zero_idx],
                                        states.r[non_zero_idx],
                                        controls.aileron[non_zero_idx],
                                        controls.rudder[non_zero_idx],
                                        constants)

        np.testing.assert_array_equal(forces[0, non_zero_idx], Cx * pressure)
        np.testing.assert_array_equal(forces[1, non_zero_idx], Cy * pressure)
        np.testing.assert_array_equal(forces[2, non_zero_idx], Cz * pressure)
        np.testing.assert_array_equal(moments[0, non_zero_idx], Mx * pressure)
        np.testing.assert_array_equal(moments[1, non_zero_idx], My * pressure)
        np.testing.assert_array_equal(moments[2, non_zero_idx], Mz * pressure)


@pytest.mark.parametrize('n_points', [1, 2])
def test_prop_forces_output_shape(n_points):
    states = random_states(n_points)
    controls = random_controls(n_points)

    va, _, _ = states.airspeed

    assert va.shape == controls.throttle.shape == (n_points,)

    thrust, torque = aero.prop_forces(va, controls.throttle, constants)

    assert thrust.shape == torque.shape == (n_points,)


def test_prop_forces_zero_rotation():
    """
    Special case where omega = 0 [rad/s]. From Beard supplement,
        thrust = rho * D_prop ** 2 * C_T2 * va ** 2
    and
        torque = rho * D_prop ** 3 * C_Q2 * va ** 2
    At the same time, this requires
        torque = KQ * (voltage / R_motor - i0)
    which implies
        va ** 2 = KQ / (rho * D_prop ** 3 * C_Q2) * (voltage / R_motor - i0)
    Since C_Q2 < 0, this is only valid for voltage <= i0 * R_motor.
    """
    max_volt = constants.i0 * constants.R_motor
    max_throttle = max_volt / constants.V_max

    throttle = np.linspace(0., max_throttle)
    voltage = throttle * constants.V_max

    va = np.sqrt(constants.KQ * (voltage / constants.R_motor - constants.i0)
                 / (constants.rho * constants.D_prop ** 3 * constants.C_Q2))

    thrust = constants.rho * constants.D_prop ** 2 * constants.C_T2 * va ** 2
    torque = constants.KQ * (voltage / constants.R_motor - constants.i0)

    comp_thrust, comp_torque = aero.prop_forces(va, throttle, constants)

    np.testing.assert_allclose(comp_thrust, thrust, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(comp_torque, torque, atol=1e-14, rtol=1e-14)
