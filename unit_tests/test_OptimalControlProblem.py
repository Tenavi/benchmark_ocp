import pytest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from optimalcontrol.problem import ProblemParameters

from . import ocp_dict

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
def test_initialize(ocp_name):
    problem = ocp_dict[ocp_name]["ocp"](dummy_variable=False)

    assert problem.n_states
    assert problem.n_controls
    assert np.isinf(problem.final_time) or problem.final_time > 0.
    assert isinstance(problem.parameters, ProblemParameters)

    assert not problem.parameters.dummy_variable
    problem.parameters.update(dummy_variable=True)
    assert problem.parameters.dummy_variable

#@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
#def test_cost_functions(ocp_name):
#    problem = ocp_dict[ocp_name]["ocp"]()
