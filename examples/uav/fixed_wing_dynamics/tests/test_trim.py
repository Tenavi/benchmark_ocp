import pytest

import numpy as np

from examples.common_utilities.dynamics import quaternion_to_euler

from examples.uav.fixed_wing_dynamics import trim
from examples.uav.vehicle_models.aerosonde import constants, aero_model


@pytest.mark.parametrize('va', np.arange(18., 33., 2.))
def test_trim_cruise(va):
    tol = 1e-02

    trim_state, trim_controls, dxdt = trim.compute_trim(va, constants,
                                                        aero_model)
    trim_pitch = quaternion_to_euler(trim_state.attitude)[1]

    # Confirm that aircraft is (nearly) in trim
    np.testing.assert_allclose(dxdt.to_array(), 0., atol=tol)

    # Confirm that desired airspeed is achieved
    va_compute, alpha, beta = trim_state.airspeed
    np.testing.assert_allclose(va_compute, va, atol=1e-07, rtol=1e-07)
    np.testing.assert_allclose(alpha, trim_pitch, atol=1e-07, rtol=1e-07)
    np.testing.assert_allclose(beta, 0., atol=1e-07)

    # Confirm that controls remain unsaturated at trim
    sat_controls = trim_controls.saturate(constants.min_controls,
                                          constants.max_controls,
                                          inplace=False)
    np.testing.assert_array_equal(trim_controls.to_array(),
                                  sat_controls.to_array())
