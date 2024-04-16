import pytest

import numpy as np

from examples.uav.dynamics_model import trim
from examples.uav.dynamics_model.parameters import aerosonde as constants


@pytest.mark.parametrize('va', np.arange(18., 33., 2.))
def test_trim_cruise(va):
    tol = 1e-02

    trim_state, trim_controls, dxdt = trim.compute_trim(va, constants)

    # Confirm that aircraft is in trim
    np.testing.assert_allclose(dxdt.to_array(), 0., atol=tol)

    # Confirm that desired airspeed is achieved
    va_compute, _, _ = trim_state.airspeed
    np.testing.assert_allclose(va_compute, va, atol=tol, rtol=tol)

    # Confirm that controls remain unsaturated at trim
    sat_controls = trim_controls.saturate(constants.min_controls,
                                          constants.max_controls,
                                          inplace=False)
    np.testing.assert_array_equal(trim_controls.to_array(),
                                  sat_controls.to_array())
