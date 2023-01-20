import pytest

import numpy as np

from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.parameters import ProblemParameters
from optimalcontrol.utilities import approx_derivative

rng = np.random.default_rng()

@pytest.mark.parametrize("n_states", [1,2])
@pytest.mark.parametrize("n_controls", [1,2])
def test_init(n_states, n_controls):
    A = rng.normal(size=(n_states, n_states))
    B = rng.normal(size=(n_states, n_controls))
    Q = rng.normal(size=(n_states, n_states))
    Q = Q.T @ Q
    R = rng.normal(size=(n_controls, n_controls))
    R = R.T @ R + 1e-08 * np.eye(n_controls)

    problem = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.)

    # Check that basic properties have been implemented
    assert problem.n_states == n_states
    assert problem.n_controls == n_controls
    assert np.isinf(problem.final_time)
    assert isinstance(problem.parameters, ProblemParameters)
    assert hasattr(problem, "_x0_sampler")
    np.testing.assert_allclose(
        np.linalg.cholesky(Q).T,
        problem._x0_sampler.norm
    )

    # Check that problem parameters can be updated
    A = rng.normal(size=(n_states, n_states))
    problem.parameters.update(A=A)
    np.testing.assert_allclose(A, problem.A)

@pytest.mark.parametrize("n_states", [1,2])
@pytest.mark.parametrize("n_controls", [1,2])
def test_bad_inits(n_states, n_controls):
    A = rng.normal(size=(n_states, n_states))
    B = rng.normal(size=(n_states, n_controls))
    Q = rng.normal(size=(n_states, n_states))
    Q = Q.T @ Q
    R = rng.normal(size=(n_controls, n_controls))
    R = R.T @ R + 1e-08 * np.eye(n_controls)

    # Missing A matrix
    with pytest.raises(RuntimeError, match="A"):
        problem = LinearQuadraticProblem(B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.)

    # Non-square A matrix
    with pytest.raises(ValueError, match="A"):
        bad_mat = rng.normal(size=(7,))
        problem = LinearQuadraticProblem(
            A=bad_mat, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.
        )

    # Missing B matrix
    with pytest.raises(RuntimeError, match="B"):
        problem = LinearQuadraticProblem(A=A, Q=Q, R=R, x0_lb=-1., x0_ub=1.)

    # B matrix of wrong size
    with pytest.raises(ValueError, match="B"):
        bad_mat = rng.normal(size=(n_states+1,n_controls))
        problem = LinearQuadraticProblem(
            A=A, B=bad_mat, Q=Q, R=R, x0_lb=-1., x0_ub=1.
        )

    # Missing Q matrix
    with pytest.raises(RuntimeError, match="Q"):
        problem = LinearQuadraticProblem(A=A, B=B, R=R, x0_lb=-1., x0_ub=1.)

    # Non-definite Q matrix
    if n_states == 1:
        bad_mats = [np.array([[-1e-14]]), rng.normal(size=(1,2))]
    else:
        bad_mats = [
            rng.normal(size=(n_states,n_states)),
            rng.normal(size=(n_states+1,n_states)),
            rng.normal(size=(n_states,1))
        ]
        for i in (1,2):
            bad_mats[i] = bad_mats[i] @ bad_mats[i].T
        bad_mats[-1] -= 1e-14 * np.eye(n_states)
    for bad_mat in bad_mats:
        with pytest.raises(ValueError, match="Q"):
            problem = LinearQuadraticProblem(
                A=A, B=B, Q=bad_mat, R=R, x0_lb=-1., x0_ub=1.
            )

    # Missing R matrix
    with pytest.raises(RuntimeError, match="R"):
        problem = LinearQuadraticProblem(A=A, B=B, Q=Q, x0_lb=-1., x0_ub=1.)

"""
@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_cost_functions(ocp_name, n_samples):
    problem = ocp_dict[ocp_name]["ocp"]()

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
        print("%s OCP has no terminal cost." % ocp_name)

    # Check that Jacobians give the correct size
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

    # Check shapes for flat vector inputs
    if n_samples == 1:
        dLdx, dLdu = problem.running_cost_gradients(x.flatten(), u.flatten())
        assert dLdx.shape == (problem.n_states,)
        assert dLdu.shape == (problem.n_controls,)

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
        dLdx, dLdu = problem.running_cost_hessians(x.flatten(), u.flatten())
        assert dLdx.shape == (problem.n_states, problem.n_states)
        assert dLdu.shape == (problem.n_controls, problem.n_controls)

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_dynamics(ocp_name, n_samples):
    problem = ocp_dict[ocp_name]["ocp"]()

    # Get some random states and controls
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(problem.n_controls, n_samples))

    # Evaluate the vector field and check that the shape is correct
    f = problem.dynamics(x, u)
    assert f.shape == (problem.n_states, n_samples)
    # Dynamics should also handle flat vector inputs
    if n_samples == 1:
        f = problem.dynamics(x.flatten(), u.flatten())
        assert f.shape == (problem.n_states,)

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
        dfdx, dfdu = problem.jacobians(x.flatten(), u.flatten())
        assert dfdx.shape == (problem.n_states, problem.n_states)
        assert dfdu.shape == (problem.n_states, problem.n_controls)

@pytest.mark.parametrize("ocp_name", ocp_dict.keys())
@pytest.mark.parametrize("n_samples", range(1,3))
def test_optimal_control(ocp_name, n_samples):
    problem = ocp_dict[ocp_name]["ocp"]()

    # Get some random states and costates
    x = problem.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(problem.n_states, n_samples)
    p = problem.sample_initial_conditions(n_samples=n_samples)
    p = p.reshape(problem.n_states, n_samples)

    # Evaluate the optimal control and check that the shape is correct
    u = problem.optimal_control(x, p)
    assert u.shape == (problem.n_controls, n_samples)
    # Optimal control should also handle flat vector inputs
    if n_samples == 1:
        u = problem.optimal_control(x.flatten(), p.flatten())
        assert u.shape == (problem.n_controls,)

    # Check that Jacobian gives the correct size
    dudx = problem.optimal_control_jacobian(x, p)
    assert dudx.shape == (problem.n_controls, problem.n_states, n_samples)

    compare_finite_difference(
        x, dudx, lambda x: problem.optimal_control(x, p),
        method=problem._fin_diff_method
    )

    # Check shape for flat vector inputs
    if n_samples == 1:
        dudx = problem.optimal_control_jacobian(x.flatten(), p.flatten())
        assert dudx.shape == (problem.n_controls, problem.n_states)
"""
