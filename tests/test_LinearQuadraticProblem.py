import numpy as np
import pytest

from optimalcontrol.problem import LinearQuadraticProblem, ProblemParameters
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params, compare_finite_difference


rng = np.random.default_rng(123)


def _make_indefinite_matrices(n, strict=True):
    """Generate random non-square and non-positive-definite cost matrices."""
    if n == 1:
        bad_mats = [np.array([[-1e-14]]),
                    rng.normal(size=(1, 2)),
                    rng.normal(size=(2, 1))]
    else:
        bad_mats = [rng.normal(size=(n, n)),
                    rng.normal(size=(n+1, n)),
                    rng.normal(size=(n, 1))]
        for i in (1, 2):
            bad_mats[i] = bad_mats[i] @ bad_mats[i].T
        bad_mats[-1] -= 1e-14 * np.eye(n)

    if strict:
        bad_mats.append(np.zeros((n, n)))

    return bad_mats


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('n_controls', [1, 2])
def test_init(n_states, n_controls):
    """Test that the LQR problem can be initialized and allows parameters to be
    updated as expected."""
    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.)

    # Check that basic properties have been implemented
    assert ocp.n_states == n_states
    assert ocp.n_controls == n_controls
    assert np.isinf(ocp.final_time)
    assert isinstance(ocp.parameters, ProblemParameters)
    np.testing.assert_allclose(np.linalg.cholesky(Q).T,
                               ocp.parameters._x0_sampler.norm, atol=1e-12)

    # Check that problem parameters can be updated
    assert not np.allclose(xf, ocp.parameters.xf)
    ocp.parameters.update(xf=xf)
    np.testing.assert_allclose(xf, ocp.parameters.xf, atol=1e-12)

    # Check that updating with nothing doesn't make any errors
    ocp.parameters.update()

    # Check that a new instance of the problem doesn't carry old parameters
    ocp2 = LinearQuadraticProblem(A=A + 1., B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.)
    np.testing.assert_allclose(ocp.parameters.A, A, atol=1e-12)
    np.testing.assert_allclose(ocp2.parameters.A, A + 1., atol=1e-12)


@pytest.mark.parametrize('missing', ['A', 'B', 'Q', 'R', 'x0_lb', 'x0_ub'])
def test_missing_inits(missing):
    """Test that initializing the LQR problem without all required parameters
    raises an exception."""
    n_states = rng.choice(range(1,10))
    n_controls = rng.choice(range(1,10))

    init_dict = dict(zip(['A', 'B', 'Q', 'R', 'xf', 'uf'],
                         make_LQ_params(n_states, n_controls)))
    init_dict.update({'x0_lb': - rng.uniform(size=n_states) - 1.,
                      'x0_ub': rng.uniform(size=n_states) + 1.})

    init_dict.pop(missing)
    with pytest.raises(RuntimeError, match=missing):
        _ = LinearQuadraticProblem(**init_dict)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('n_controls', [1, 2])
def test_bad_inits(n_states, n_controls):
    """Test that initializing the LQR problem with matrices of incorrect size or
    indefinite matrices raises an exception."""
    init_dict = dict(zip(['A', 'B', 'Q', 'R', 'xf', 'uf'],
                         make_LQ_params(n_states, n_controls)))
    init_dict.update({'x0_lb': -1., 'x0_ub': 1.})

    # Non-square A matrix
    bad_sizes = [(1, 2), (2, 3), (2, 1), (3, 2), (2,), (3,)]
    for bad_size in bad_sizes:
        bad_init = {**init_dict, 'A': rng.normal(size=bad_size)}
        with pytest.raises(ValueError, match='A'):
            _ = LinearQuadraticProblem(**bad_init)

    # B matrix of wrong size
    bad_init = {**init_dict, 'B': rng.normal(size=(n_states+1, n_controls))}
    with pytest.raises(ValueError, match='B'):
        _ = LinearQuadraticProblem(**bad_init)

    # Non positive semi-definite Q matrix
    for bad_mat in _make_indefinite_matrices(n_states, strict=False):
        bad_init = {**init_dict, 'Q': bad_mat}
        with pytest.raises(ValueError, match="Q"):
            _ = LinearQuadraticProblem(**bad_init)

    # Non positive-definite R matrix
    for bad_mat in _make_indefinite_matrices(n_states, strict=True):
        bad_init = {**init_dict, 'R': bad_mat}
        with pytest.raises(ValueError, match="R"):
            _ = LinearQuadraticProblem(**bad_init)


@pytest.mark.parametrize('n_states', [1, 2])
def test_sample(n_states):
    """Test that we can sample initial conditions from the LQ problem and the
    distance matrix can be updated."""
    n_controls = rng.choice(range(1, 10))
    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    x0_lb = - rng.uniform(size=(n_states, 1)) - 1.
    x0_ub = rng.uniform(size=(n_states, 1)) + 1.
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=x0_lb, x0_ub=x0_ub,
                                 xf=xf, uf=uf)

    # Without distance specification
    for n_samples in range(1, 5):
        x0 = ocp.sample_initial_conditions(n_samples)
        if n_samples > 1:
            assert x0.shape == (n_states, n_samples)
        else:
            assert x0.shape == (n_states,)
            x0 = x0.reshape(n_states, 1)
        assert np.all(x0_lb <= x0)
        assert np.all(x0 <= x0_ub)

    # With distance specification
    def check_distance(n_samples, distance, rtol=1e-06, atol=1e-06):
        x0 = ocp.sample_initial_conditions(n_samples, distance=distance)
        if n_samples > 1:
            assert x0.shape == (n_states, n_samples)
        else:
            assert x0.shape == (n_states,)
            x0 = x0.reshape(n_states, 1)
        xQx = ocp.distances(x0, xf)
        np.testing.assert_allclose(distance, xQx, rtol=rtol, atol=atol)

    distances = rng.uniform(size=(2,)) + np.array([0, 1])
    for distance in distances:
        for n_samples in range(1, 10):
            check_distance(n_samples, distance)

    # Check again after updating Q matrix
    Q = Q + np.eye(n_states)
    assert not np.allclose(Q, ocp.parameters.Q)
    ocp.parameters.update(Q=Q)
    np.testing.assert_allclose(Q, ocp.parameters.Q, atol=1e-12)

    for distance in distances:
        for n_samples in range(1, 10):
            check_distance(n_samples, distance)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('n_controls', [1, 2])
@pytest.mark.parametrize('n_samples', [1, 50])
def test_cost_functions(n_states, n_controls, n_samples):
    """Test that cost function inputs and outputs have the correct shape and
    that gradients and Hessians of return the expected Q and R matrices (except
    when the control is saturated, in which case the corresponding parts of R
    should be zero)."""
    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.,
                                 xf=xf, uf=uf, u_lb=-0.5)

    # Get some random states and controls. Some controls will be saturated.
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(ocp.n_controls, n_samples))

    # Evaluate the cost functions and check that the shapes are correct
    L = ocp.running_cost(x, u)
    assert L.ndim == 1
    assert L.shape[0] == n_samples

    # Check that gradients give the correct size
    dLdx, dLdu = ocp.running_cost_grad(x, u)
    assert dLdx.shape == (n_states, n_samples)
    assert dLdu.shape == (n_controls, n_samples)

    # Check that control gradients match with finite difference approximation
    fin_diff_dLdu = (super(LinearQuadraticProblem, ocp)
                     .running_cost_grad(x, u, return_dLdx=False))
    np.testing.assert_allclose(dLdu, fin_diff_dLdu, atol=1e-06)

    # Check that Hessians are the correct size
    dLdx2, dLdu2 = ocp.running_cost_hess(x, u)
    assert dLdx2.shape == (n_states, n_states, n_samples)
    assert dLdu2.shape == (n_controls, n_controls, n_samples)

    # Check that control Hessians match with finite difference approximation
    fin_diff_dLdu2 = (super(LinearQuadraticProblem, ocp)
                      .running_cost_hess(x, u, return_dLdx=False))
    np.testing.assert_allclose(dLdu2, fin_diff_dLdu2, atol=1e-06)

    # Check that vectorized construction matches brute force
    for i in range(n_samples):
        xi = x[:, i] - xf.flatten()
        ui = u[:, i] - uf.flatten()

        np.testing.assert_allclose(L[i], xi @ Q @ xi + ui @ R @ ui, atol=1e-12)

        np.testing.assert_allclose(dLdx[..., i], 2. * Q @ xi, atol=1e-12)
        np.testing.assert_allclose(dLdx2[..., i], Q, atol=1e-12)

    # Check shapes for flat vector inputs
    if n_samples == 1:
        L = ocp.running_cost(x.flatten(), u.flatten())
        assert L.ndim == 0

        dLdx, dLdu = ocp.running_cost_grad(x.flatten(), u.flatten())
        assert dLdx.shape == (n_states,)
        assert dLdu.shape == (n_controls,)

        dLdx, dLdu = ocp.running_cost_hess(x.flatten(), u.flatten())
        assert dLdx.shape == (n_states, n_states)
        assert dLdu.shape == (n_controls, n_controls)


@pytest.mark.parametrize('n_states', [1, 2])
@pytest.mark.parametrize('n_controls', [1, 2])
@pytest.mark.parametrize('n_samples', [1, 10])
def test_dynamics(n_states, n_controls, n_samples):
    """Test that dynamic inputs and outputs have the correct shape and that the
    Jacobians return the expected matrices (except when the control is
    saturated, in which case the corresponding parts should be zero)."""
    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.,
                                 xf=xf, uf=uf, u_lb=-0.5, u_ub=0.5)

    # Get some random states and controls. Some controls will be saturated.
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    u = rng.uniform(low=-1., high=1., size=(ocp.n_controls, n_samples))

    # Evaluate the vector field and check that the shape is correct
    f = ocp.dynamics(x, u)
    assert f.shape == (n_states, n_samples)

    # Check that Jacobians give the correct size
    dfdx, dfdu = ocp.jac(x, u)
    assert dfdx.shape == (n_states, n_states, n_samples)
    assert dfdu.shape == (n_states, n_controls, n_samples)

    # Check that control Jacobians match with finite difference approximation,
    # which is equal to B when the control is unsaturated and zero when
    # saturated.
    fin_diff_dfdu = (
        super(LinearQuadraticProblem, ocp).jac(x, u, return_dfdx=False))
    np.testing.assert_allclose(dfdu, fin_diff_dfdu, rtol=1e-06, atol=1e-12)

    # Check that vectorized construction matches brute force
    for i in range(n_samples):
        xi = x[:, i] - xf.flatten()
        ui = u[:, i] - uf.flatten()

        np.testing.assert_allclose(f[:, i], A @ xi + B @ ui, atol=1e-12)

        np.testing.assert_allclose(dfdx[..., i], A, atol=1e-12)

    # Check shapes for flat vector inputs
    if n_samples == 1:
        f = ocp.dynamics(x.flatten(), u.flatten())
        assert f.shape == (n_states,)

        dfdx, dfdu = ocp.jac(x.flatten(), u.flatten())
        assert dfdx.shape == (n_states, n_states)
        assert dfdu.shape == (n_states, n_controls)


@pytest.mark.parametrize('n_states', (1, 3))
@pytest.mark.parametrize('n_controls', (1, 2))
@pytest.mark.parametrize('n_samples', (1, 4))
@pytest.mark.parametrize('u_ub', (0.5, None))
def test_hamiltonian(n_states, n_controls, n_samples, u_ub):
    """Test that the optimal control as a function of state and costate matches
    LQR when costates are optimal."""
    u_lb = None if u_ub is None else -u_ub

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=123)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, x0_lb=-1., x0_ub=1.,
                                 xf=xf, uf=uf, u_lb=u_lb, u_ub=u_ub,
                                 x0_sample_seed=456)

    # Get some random states and costates
    x = ocp.sample_initial_conditions(n_samples=n_samples)
    x = x.reshape(ocp.n_states, n_samples)
    p = ocp.sample_initial_conditions(n_samples=n_samples)
    p = p.reshape(ocp.n_states, n_samples)

    # Evaluate the optimal control and check shape
    # (the numerical optimizer method works badly for the LQR problem, and we
    # will check it later against LQR)
    u = ocp.hamiltonian_minimizer(x, p)
    assert u.shape == (n_controls, n_samples)

    # Check that Jacobian gives the correct size
    dudx = ocp.hamiltonian_minimizer_jac(x, p)
    assert dudx.shape == (n_controls, n_states, n_samples)

    compare_finite_difference(x, dudx,
                              lambda x: ocp.hamiltonian_minimizer(x, p),
                              method=ocp._fin_diff_method)

    # Check shape for flat vector inputs
    if n_samples == 1:
        u = ocp.hamiltonian_minimizer(x.flatten(), p.flatten())
        assert u.shape == (n_controls,)

        dudx = ocp.hamiltonian_minimizer_jac(x.flatten(), p.flatten())
        assert dudx.shape == (n_controls, n_states)

    # Compare with LQR solution
    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                   u_lb=u_lb, u_ub=u_ub)

    p = 2. * lqr.P @ (x - xf)
    u = ocp.hamiltonian_minimizer(x, p)
    u_expected = lqr(x)

    np.testing.assert_allclose(u, u_expected, rtol=1e-06, atol=1e-12)
