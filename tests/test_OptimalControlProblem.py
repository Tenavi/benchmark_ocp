import pytest

import numpy as np

from optimalcontrol.parameters import ProblemParameters

from ._problems import ocp_dict
from ._utilities import compare_finite_difference

rng = np.random.default_rng()

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
def test_init(ocp_name):
    """Basic check that each OCP can be initialized and allows parameters to be
    updated as expected."""
    problem = ocp_dict[ocp_name]["ocp"]()

    # Check that basic properties have been implemented
    assert problem.n_states
    assert problem.n_controls
    assert np.isinf(problem.final_time) or problem.final_time > 0.
    assert isinstance(problem.parameters, ProblemParameters)

    for param in problem.parameters.required:
        assert getattr(problem.parameters, param) is not None

    # Check that problem parameters can be updated
    problem.parameters.optional = {"dummy_variable": False}
    assert not problem.parameters.dummy_variable
    problem.parameters.update(dummy_variable=True)
    assert problem.parameters.dummy_variable

    # Check that updating with nothing doesn't make any errors
    problem.parameters.update()

    # Check that a new instance of the problem doesn't carry old parameters
    problem2 = ocp_dict[ocp_name]["ocp"]()
    assert not hasattr(problem2.parameters, "dummy_variable")

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_sample_initial_conditions(ocp_name, n_samples):
    """Test that we can sample initial conditions from each OCP."""
    problem = ocp_dict[ocp_name]["ocp"]()

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

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_cost_functions(ocp_name, n_samples):
    """Test that cost function inputs and outputs have the correct shape and
    that gradients and Hessian of the cost function match finite difference
    approximations."""
    problem = ocp_dict[ocp_name]["ocp"]()

    # Get some random states and controls
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(problem.n_controls, n_samples))

    # Evaluate the cost functions and check that the shapes are correct
    L = problem.running_cost(x, u)
    assert L.ndim == 1
    assert L.shape[0] == n_samples

    try:
        F = problem.terminal_cost(x)
        assert F.ndim == 1
        assert F.shape[0] == n_samples
        # Check shapes for flat vector inputs
        if n_samples == 1:
            F = problem.terminal_cost(x.flatten())
            assert F.ndim == 0
    except NotImplementedError:
        print("%s OCP has no terminal cost." % ocp_name)

    # Check that gradients give the correct size
    dLdx, dLdu = problem.running_cost_gradients(x, u)
    assert dLdx.shape == (problem.n_states, n_samples)
    assert dLdu.shape == (problem.n_controls, n_samples)

    compare_finite_difference(
        x, dLdx, lambda x: problem.running_cost(x, u),
        method=problem._fin_diff_method
    )
    compare_finite_difference(
        u, dLdu, lambda u: problem.running_cost(x, u),
        method=problem._fin_diff_method
    )

    # Check that Hessians give the correct size
    dLdx, dLdu = problem.running_cost_hessians(x, u)
    assert dLdx.shape == (problem.n_states, problem.n_states, n_samples)
    assert dLdu.shape == (problem.n_controls, problem.n_controls, n_samples)

    compare_finite_difference(
        x, dLdx,
        lambda x: problem.running_cost_gradients(x, u, return_dLdu=False),
        method=problem._fin_diff_method
    )
    compare_finite_difference(
        u, dLdu,
        lambda u: problem.running_cost_gradients(x, u, return_dLdx=False),
        method=problem._fin_diff_method
    )

    # Check shapes for flat vector inputs
    if n_samples == 1:
        L = problem.running_cost(x.flatten(), u.flatten())
        assert L.ndim == 0

        dLdx, dLdu = problem.running_cost_gradients(x.flatten(), u.flatten())
        assert dLdx.shape == (problem.n_states,)
        assert dLdu.shape == (problem.n_controls,)

        dLdx, dLdu = problem.running_cost_hessians(x.flatten(), u.flatten())
        assert dLdx.shape == (problem.n_states, problem.n_states)
        assert dLdu.shape == (problem.n_controls, problem.n_controls)

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_dynamics(ocp_name, n_samples):
    """Test that dynamics inputs and outputs have the correct shape and that
    Jacobians match finite difference approximations."""
    problem = ocp_dict[ocp_name]["ocp"]()

    # Get some random states and controls
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(problem.n_controls, n_samples))

    # Evaluate the vector field and check that the shape is correct
    f = problem.dynamics(x, u)
    assert f.shape == (problem.n_states, n_samples)

    # Check that Jacobians give the correct size
    dfdx, dfdu = problem.jacobians(x, u)
    assert dfdx.shape == (problem.n_states, problem.n_states, n_samples)
    assert dfdu.shape == (problem.n_states, problem.n_controls, n_samples)

    compare_finite_difference(
        x, dfdx, lambda x: problem.dynamics(x, u),
        method=problem._fin_diff_method)
    compare_finite_difference(
        u, dfdu, lambda u: problem.dynamics(x, u),
        method=problem._fin_diff_method
    )

    # Check shapes for flat vector inputs
    if n_samples == 1:
        f = problem.dynamics(x.flatten(), u.flatten())
        assert f.shape == (problem.n_states,)

        dfdx, dfdu = problem.jacobians(x.flatten(), u.flatten())
        assert dfdx.shape == (problem.n_states, problem.n_states)
        assert dfdu.shape == (problem.n_states, problem.n_controls)

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_optimal_control(ocp_name, n_samples):
    """Test that the optimal control as a function of state and costate returns
    the correct shape and Jacobians match finite difference approximations."""
    problem = ocp_dict[ocp_name]["ocp"]()

    # Get some random states and costates
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    p = problem.sample_initial_conditions(n_samples=n_samples)
    p = p.reshape(problem.n_states, n_samples)

    # Evaluate the optimal control and check that the shape is correct
    u = problem.optimal_control(x, p)
    assert u.shape == (problem.n_controls, n_samples)

    # Check that Jacobian gives the correct size
    dudx = problem.optimal_control_jacobian(x, p)
    assert dudx.shape == (problem.n_controls, problem.n_states, n_samples)

    compare_finite_difference(
        x, dudx, lambda x: problem.optimal_control(x, p),
        method=problem._fin_diff_method
    )

    # Check shape for flat vector inputs
    if n_samples == 1:
        u = problem.optimal_control(x.flatten(), p.flatten())
        assert u.shape == (problem.n_controls,)

        dudx = problem.optimal_control_jacobian(x.flatten(), p.flatten())
        assert dudx.shape == (problem.n_controls, problem.n_states)
