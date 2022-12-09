import pytest

import numpy as np

from optimalcontrol.problem import ProblemParameters
from optimalcontrol.utilities import approx_derivative

ocp_dict = {}

from example_problems.van_der_pol import van_der_pol
ocp_dict['van_der_pol'] = {
    'ocp': van_der_pol.VanDerPol, 'config': van_der_pol.config
}

rng = np.random.default_rng()

def _check_finite_differences(x, jac, fun):
    expected_jac = approx_derivative(fun, x)
    np.testing.assert_allclose(jac, expected_jac)

@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
def test_init(ocp_name):
    problem = ocp_dict[ocp_name]['ocp'](dummy_variable=False)

    # Check that basic properties have been implemented
    assert problem.n_states
    assert problem.n_controls
    assert np.isinf(problem.final_time) or problem.final_time > 0.
    assert isinstance(problem.parameters, ProblemParameters)

    # Check that problem parameters can be updated
    assert not problem.parameters.dummy_variable
    problem.parameters.update(dummy_variable=True)
    assert problem.parameters.dummy_variable

@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', range(1,3))
def test_sample_initial_conditions(ocp_name, n_samples):
    problem = ocp_dict[ocp_name]['ocp']()

    with pytest.raises(Exception):
        problem.sample_initial_conditions(n_samples=0)

    # Check that sample_initial_conditions returns the correct size arrays
    x0 = problem.sample_initial_conditions(n_samples=n_samples)

    if n_samples == 1:
        assert x0.ndim == 1
        x0 = x0.reshape(-1,1)
    else:
        assert x0.ndim == 2
        assert x0.shape[1] == n_samples
    assert x0.shape[0] == problem.n_states

@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', range(1,3))
def test_cost_functions(ocp_name, n_samples):
    problem = ocp_dict[ocp_name]['ocp']()

    # Get some random states and controls
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(problem.n_controls, n_samples))

    # Evaluate the cost functions and check that the shapes are correct
    L = problem.running_cost(x, u)
    assert L.ndim == 1
    assert L.shape[0] == n_samples
    # Cost functions should also handle flat vector inputs
    if n_samples == 1:
        L = problem.running_cost(x.flatten(), u.flatten())
        assert L.ndim == 0

    try:
        F = problem.terminal_cost(x)
        assert F.ndim == 1
        assert F.shape[0] == n_samples
        # Check shapes for flat vector inputs
        if n_samples == 1:
            F = problem.terminal_cost(x.flatten())
            assert F.ndim == 0
    except NotImplementedError:
        print('%s OCP has no terminal cost.' % ocp_name)

    # Check that Jacobians give the correct size
    dLdx, dLdu = problem.running_cost_gradients(x, u)
    assert dLdx.shape == (problem.n_states, n_samples)
    assert dLdu.shape == (problem.n_controls, n_samples)

    _check_finite_differences(x, dLdx, lambda x: problem.running_cost(x, u))
    _check_finite_differences(u, dLdu, lambda u: problem.running_cost(x, u))

    # Check shapes for flat vector inputs
    if n_samples == 1:
        dLdx, dLdu = problem.running_cost_gradients(x.flatten(), u.flatten())
        assert dLdx.shape == (problem.n_states,)
        assert dLdu.shape == (problem.n_controls,)

    # Check that Hessians give the correct size
    '''dLdx, dLdu = problem.running_cost_hessians(x, u)
    assert dLdx.ndim == dLdu.ndim == 3
    assert dLdx.shape[-1] == dLdu.shape[-1] == n_samples
    assert dLdx.shape[0] == dLdx.shape[1] == problem.n_states
    assert dLdu.shape[0] == dLdu.shape[1] == problem.n_controls

    _check_finite_differences(
        x, u, problem.running_cost_gradients, dLdx, dLdu
    )

    # Check shapes for flat vector inputs
    if False:#n_samples == 1:
        dLdx, dLdu = problem.running_cost_gradients(x.flatten(), u.flatten())
        assert dLdx.ndim == dLdu.ndim == 1
        assert dLdx.shape[0] == problem.n_states
        assert dLdu.shape[0] == problem.n_controls'''
