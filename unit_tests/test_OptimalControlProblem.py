import pytest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.problem import ProblemParameters

ocp_dict = {}

from examples.van_der_pol import van_der_pol
ocp_dict['van_der_pol'] = {
    'ocp': van_der_pol.VanDerPol, 'config': van_der_pol.config
}

@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
def test_initialize(ocp_name):
    problem = ocp_dict[ocp_name]['ocp'](dummy_variable=False)

    # Check that basic properties have been implemented
    assert problem.n_states
    assert problem.n_controls
    assert np.isinf(problem.final_time) or problem.final_time > 0.
    assert isinstance(problem.parameters, ProblemParameters)

    # Check that problem parameters are updateable
    assert not problem.parameters.dummy_variable
    problem.parameters.update(dummy_variable=True)
    assert problem.parameters.dummy_variable

    # Check that sample_initial_conditions returns the correct size arrays
    with pytest.raises(Exception):
        problem.sample_initial_conditions(n_samples=0)
    x0 = problem.sample_initial_conditions(n_samples=1)
    assert x0.ndim == 1
    assert x0.shape[0] == problem.n_states
    for n_samples in range(2,4):
        x0 = problem.sample_initial_conditions(n_samples=n_samples)
        assert x0.ndim == 2
        assert x0.shape[0] == problem.n_states
        assert x0.shape[1] == n_samples

#@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
#def test_cost_functions(ocp_name):
#    problem = ocp_dict[ocp_name]['ocp']()
