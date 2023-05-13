import pytest

import numpy as np

from optimalcontrol.problem import ProblemParameters

from ._problems import ocp_dict
from ._utilities import compare_finite_difference


rng = np.random.default_rng()


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
def test_init(ocp_name):
    """Basic check that each OCP can be initialized and allows parameters to be
    updated as expected."""
    ocp = ocp_dict[ocp_name]()

    # Check that basic properties have been implemented
    assert ocp.n_states
    assert ocp.n_controls
    assert np.isinf(ocp.final_time) or ocp.final_time > 0.
    assert isinstance(ocp.parameters, ProblemParameters)

    for param in ocp.parameters.required:
        assert getattr(ocp.parameters, param) is not None

    # Check that problem parameters can be updated
    ocp.parameters.optional = {'dummy_variable': False}
    assert not ocp.parameters.dummy_variable
    ocp.parameters.update(dummy_variable=True)
    assert ocp.parameters.dummy_variable

    # Check that updating with nothing doesn't make any errors
    ocp.parameters.update()

    # Check that a new instance of the problem doesn't carry old parameters
    ocp2 = ocp_dict[ocp_name]()
    assert not hasattr(ocp2.parameters, 'dummy_variable')


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', [1, 2])
def test_sample_initial_conditions(ocp_name, n_samples):
    """Test that we can sample initial conditions from each OCP."""
    ocp = ocp_dict[ocp_name]()

    with pytest.raises(Exception):
        ocp.sample_initial_conditions(n_samples=0)

    # Check that sample_initial_conditions returns the correct size arrays
    x0 = ocp.sample_initial_conditions(n_samples=n_samples)

    if n_samples == 1:
        assert x0.ndim == 1
        x0 = x0.reshape(-1,1)
    else:
        assert x0.ndim == 2
        assert x0.shape[1] == n_samples
    assert x0.shape[0] == ocp.n_states


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', [1, 2])
def test_cost_functions(ocp_name, n_samples):
    """Test that cost function inputs and outputs have the correct shape and
    that gradients and Hessian of the cost function match finite difference
    approximations."""
    ocp = ocp_dict[ocp_name]()

    # Get some random states and controls
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    u = rng.uniform(low=-2., high=2., size=(ocp.n_controls, n_samples))

    # Evaluate the cost functions and check that the shapes are correct
    L = ocp.running_cost(x, u)
    assert L.ndim == 1
    assert L.shape[0] == n_samples

    try:
        F = ocp.terminal_cost(x)
        assert F.ndim == 1
        assert F.shape[0] == n_samples
        # Check shapes for flat vector inputs
        if n_samples == 1:
            F = ocp.terminal_cost(x.flatten())
            assert F.ndim == 0
    except NotImplementedError:
        print(f'{ocp_name} OCP has no terminal cost.')

    # Check that gradients give the correct size
    dLdx, dLdu = ocp.running_cost_grad(x, u)
    assert dLdx.shape == (ocp.n_states, n_samples)
    assert dLdu.shape == (ocp.n_controls, n_samples)

    compare_finite_difference(x, dLdx, lambda x: ocp.running_cost(x, u),
                              method=ocp._fin_diff_method, atol=1e-05)
    compare_finite_difference(u, dLdu, lambda u: ocp.running_cost(x, u),
                              method=ocp._fin_diff_method, atol=1e-05)

    # Check that Hessians are the correct size
    dLdx, dLdu = ocp.running_cost_hess(x, u)
    assert dLdx.shape == (ocp.n_states, ocp.n_states, n_samples)
    assert dLdu.shape == (ocp.n_controls, ocp.n_controls, n_samples)

    compare_finite_difference(
        x, 2. * dLdx,
        lambda x: ocp.running_cost_grad(x, u, return_dLdu=False),
        method=ocp._fin_diff_method, atol=1e-05)
    compare_finite_difference(
        u, 2. * dLdu,
        lambda u: ocp.running_cost_grad(x, u, return_dLdx=False),
        method=ocp._fin_diff_method, atol=1e-05)

    # Check shapes for flat vector inputs
    if n_samples == 1:
        L = ocp.running_cost(x.flatten(), u.flatten())
        assert L.ndim == 0

        dLdx, dLdu = ocp.running_cost_grad(x.flatten(), u.flatten())
        assert dLdx.shape == (ocp.n_states,)
        assert dLdu.shape == (ocp.n_controls,)

        dLdx, dLdu = ocp.running_cost_hess(x.flatten(), u.flatten())
        assert dLdx.shape == (ocp.n_states, ocp.n_states)
        assert dLdu.shape == (ocp.n_controls, ocp.n_controls)


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', [1, 2])
def test_dynamics(ocp_name, n_samples):
    """Test that dynamics inputs and outputs have the correct shape and that
    Jacobians match finite difference approximations."""
    ocp = ocp_dict[ocp_name]()

    # Get some random states and controls
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(ocp.n_controls, n_samples))

    # Evaluate the vector field and check that the shape is correct
    f = ocp.dynamics(x, u)
    assert f.shape == (ocp.n_states, n_samples)

    # Check that Jacobians give the correct size
    dfdx, dfdu = ocp.jac(x, u)
    assert dfdx.shape == (ocp.n_states, ocp.n_states, n_samples)
    assert dfdu.shape == (ocp.n_states, ocp.n_controls, n_samples)

    compare_finite_difference(x, dfdx, lambda x: ocp.dynamics(x, u),
                              method=ocp._fin_diff_method, atol=1e-05)
    compare_finite_difference(u, dfdu, lambda u: ocp.dynamics(x, u),
                              method=ocp._fin_diff_method, atol=1e-05)

    # Check shapes for flat vector inputs
    if n_samples == 1:
        f = ocp.dynamics(x.flatten(), u.flatten())
        assert f.shape == (ocp.n_states,)

        dfdx, dfdu = ocp.jac(x.flatten(), u.flatten())
        assert dfdx.shape == (ocp.n_states, ocp.n_states)
        assert dfdu.shape == (ocp.n_states, ocp.n_controls)


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', [1, 2])
def test_optimal_control(ocp_name, n_samples):
    """Test that the optimal control as a function of state and costate returns
    the correct shape and Jacobians match finite difference approximations."""
    ocp = ocp_dict[ocp_name]()

    # Get some random states and costates
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    p = ocp.sample_initial_conditions(n_samples=n_samples)
    p = p.reshape(ocp.n_states, n_samples)

    # Evaluate the optimal control and check that the shape is correct
    u = ocp.optimal_control(x, p)
    assert u.shape == (ocp.n_controls, n_samples)

    # Check that Jacobian is the correct size
    dudx = ocp.optimal_control_jac(x, p)
    assert dudx.shape == (ocp.n_controls, ocp.n_states, n_samples)

    compare_finite_difference(x, dudx, lambda x: ocp.optimal_control(x, p),
                              method=ocp._fin_diff_method, atol=1e-05)

    # Check shape for flat vector inputs
    if n_samples == 1:
        u = ocp.optimal_control(x.flatten(), p.flatten())
        assert u.shape == (ocp.n_controls,)

        dudx = ocp.optimal_control_jac(x.flatten(), p.flatten())
        assert dudx.shape == (ocp.n_controls, ocp.n_states)


@pytest.mark.parametrize('ocp_name', ocp_dict.keys())
@pytest.mark.parametrize('n_samples', [1, 2])
def test_bvp_dynamics(ocp_name, n_samples):
    """Test that BVP dynamics inputs and outputs have the correct shape and that
    the output is (approximately) the same as the superclass method."""
    ocp = ocp_dict[ocp_name]()

    # Get some random states and controls
    t = np.linspace(0., 1., n_samples)
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    p = ocp.sample_initial_conditions(n_samples=n_samples)
    p = p.reshape(ocp.n_states, n_samples)
    v = np.exp(-t)

    xp = np.vstack((x, p, v))

    # Evaluate the vector field and check that the shape is correct
    f = ocp.bvp_dynamics(t, xp)
    assert f.shape == (2*ocp.n_states+1, n_samples)
    f_super = super(type(ocp), ocp).bvp_dynamics(t, xp)
    np.testing.assert_allclose(f, f_super, atol=1e-05)
