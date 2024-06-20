import pytest

import numpy as np

from examples.uav.fixed_wing_dynamics import trim
from examples.uav.vehicle_models.aerosonde import constants, aero_model


@pytest.mark.parametrize('va', np.arange(18., 33., 2.))
def test_trim_cruise(va):
    tol = 1e-02

    trim_state, trim_controls, dxdt = trim.compute_trim(va, constants,
                                                        aero_model)

    # Confirm that aircraft is (nearly) in trim
    np.testing.assert_allclose(dxdt.to_array(), 0., atol=tol)

    # Confirm that desired airspeed is achieved
    va_compute, _, _ = trim_state.airspeed
    np.testing.assert_allclose(va_compute, va, atol=tol, rtol=tol)

    # Confirm that controls remain unsaturated at trim
    sat_controls = trim_controls.saturate(constants.min_controls,
                                          constants.max_controls,
                                          inplace=False)
    assert trim_controls == sat_controls
