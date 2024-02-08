import numpy as np
import pytest

from optimalcontrol.utilities import approx_derivative

from examples.uav.dynamics_model import constants, aero


rng = np.random.default_rng()


@pytest.mark.parametrize('n_points', [1, 2])
def test_blending_fun(n_points):
    a0, m = constants.alpha_stall, constants.aero_blend_rate

    alpha = rng.normal(scale=a0, size=(n_points,))

    sigma = aero.blending_fun(alpha, a0)

    num = 1. + np.exp(-m * (alpha - a0)) + np.exp(m * (alpha + a0))
    den = (1. + np.exp(-m * (alpha - a0))) * (1. + np.exp(m * (alpha + a0)))
    sigma_expect = num / den

    np.testing.assert_allclose(sigma, sigma_expect, atol=1e-14)


@pytest.mark.parametrize('n_points', [1, 2])
def test_blending_fun_jac(n_points):
    a0, m = constants.alpha_stall, constants.aero_blend_rate

    alpha = rng.normal(scale=a0, size=(n_points,))

    _, sigma_d = aero.blending_fun(alpha, a0, aero_blend_rate=m, jac=True)

    sigma_d_expect = approx_derivative(lambda a: aero.blending_fun(a, a0),
                                       alpha.reshape(1, -1))
    sigma_d_expect = np.squeeze(sigma_d_expect)

    np.testing.assert_allclose(sigma_d, sigma_d_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_small_alpha(n_points):
    """Test that the aero coefficients are approximately linear (quadratic in
    the case of CD) close to alpha = 0."""
    alpha = rng.normal(scale=1e-07, size=(n_points,))

    CL, CD, CM = aero.coefs_alpha(alpha, constants)

    CL_expect = constants.CL0 + constants.CLalpha * alpha

    CD_expect = np.pi * constants.e * constants.b ** 2 / constants.S
    CD_expect = constants.CD0 + (CL_expect ** 2 / CD_expect)

    CM_expect = constants.CM0 + constants.CMalpha * alpha

    np.testing.assert_allclose(CL, CL_expect, atol=1e-07)
    np.testing.assert_allclose(CD, CD_expect, atol=1e-07)
    np.testing.assert_allclose(CM, CM_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_high_alpha(n_points):
    """Test that the aero coefficients are approximately equal to the flat plate
    model past stall."""
    a0 = constants.alpha_stall
    alpha = rng.uniform(low=2. * a0, high=np.pi, size=(n_points,))

    CL, CD, CM = aero.coefs_alpha(alpha, constants)

    CL_expect = 2. * np.sign(alpha) * (np.sin(alpha) ** 2 * np.cos(alpha))

    CD_expect = 2. * np.sin(alpha) ** 2

    CM_expect = constants.CM0 + constants.CMalpha * alpha

    np.testing.assert_allclose(CL, CL_expect, atol=1e-07)
    np.testing.assert_allclose(CD, CD_expect, atol=1e-07)
    np.testing.assert_allclose(CM, CM_expect, atol=1e-07)


@pytest.mark.parametrize('n_points', [1, 2])
def test_coefs_alpha_jac(n_points):
    alpha = rng.uniform(low=-np.pi, high=np.pi, size=(n_points,))

    _, _, _, d_CL, d_CD, d_CM = aero.coefs_alpha(alpha, constants, jac=True)

    assert d_CL.shape == d_CD.shape == d_CM.shape == alpha.shape

    def coef_wrapper(alpha):
        CL, CD, CM = aero.coefs_alpha(alpha, constants)
        return np.stack([CL, CD, CM], axis=0)

    for i in range(n_points):
        d_coef_expect = np.squeeze(approx_derivative(coef_wrapper, alpha[i]))
        np.testing.assert_allclose(d_CL[i], d_coef_expect[0], atol=1e-07)
        np.testing.assert_allclose(d_CD[i], d_coef_expect[1], atol=1e-07)
        np.testing.assert_allclose(d_CM[i], d_coef_expect[2], atol=1e-07)


def _test_aero_zero_airspeed():
    idx = [5,7]
    u, v, w, p, q, r = np.random.randn(6,10)
    u[idx], v[idx], w[idx] = 0., 0., 0.

    yaw, pitch, roll = 2.*np.pi*(np.random.rand(3,10) - .5)
    attitude = euler_to_quat(yaw, pitch/2., roll)
    states = containers.VehicleState(
        attitude=attitude, u=u, v=v, w=w, p=p, q=q, r=r
    )

    U = np.random.rand(4,10)
    controls = containers.Controls(U)

    forces, moments = dynamics.aero_forces(states, controls)

    assert np.allclose(forces[:,idx], 0.)
    assert np.allclose(moments[:,idx], 0.)
