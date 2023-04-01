import pytest
import numpy as np
import warnings
import time

from optimalcontrol import open_loop
from optimalcontrol.simulate import integrate_closed_loop
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params


try:
    import pylgr
    methods = ['indirect', 'direct']
except ImportError:
    methods = ['indirect']
    warnings.warn(
        'Could not import pylgr library. Not testing open_loop.direct.',
        ImportWarning)


@pytest.mark.parametrize('method', methods)
def test_solve_infinite_horizon(method):
    """
    Basic test of an LQR-controlled linear system. The OCP is solved over an
    approximate infinite horizon and compared with LQR, which is known to be
    optimal.
    """
    n_states = 3
    n_controls = 2
    t1_tol = 1e-14

    # Direct method is much less accurate than indirect, but the solution is
    # still considered reasonable.
    if method == 'direct':
        kwargs = {'max_nodes': 100, 'tol': 10*t1_tol}
        tol = 1e-02
    elif method == 'indirect':
        kwargs = {'max_nodes': 1000}
        tol = 1e-06

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=123)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                 x0_lb=-1., x0_ub=1., u_lb=-1., u_ub=1.,
                                 x0_sample_seed=456)
    LQR = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                   u_lb=-1., u_ub=1.)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    # Integrate over initially short time horizon
    t_span = [0., 10.]
    t, x, _ = integrate_closed_loop(ocp, LQR, t_span, x0)
    p = 2. * LQR.P @ (x - xf)
    u = LQR(x)
    v = np.einsum('ij,ij->j', x - xf, LQR.P @ (x - xf))

    start_time = time.time()

    ocp_sol = open_loop.solve_infinite_horizon(
        ocp, t, x, u=u, p=p, method=method, t1_tol=t1_tol, **kwargs)

    comp_time = time.time() - start_time
    print(f'Solution time with {method} method: {comp_time:1.1f} sec')

    assert ocp_sol.status == 0
    assert ocp_sol.t[-1] > t_span[-1]
    L = ocp.running_cost(ocp_sol.x, ocp_sol.u)
    assert L[-1] <= t1_tol

    # Verify that BVP solution matches LQR solution, which is optimal
    u_expect = LQR(ocp_sol.x)
    np.testing.assert_allclose(ocp_sol.u, u_expect, atol=tol, rtol=tol)
    p_expect = 2. * LQR.P @ (ocp_sol.x - xf)
    np.testing.assert_allclose(ocp_sol.p, p_expect, atol=tol, rtol=tol)
    v_expect = np.einsum('ij,ij->j', ocp_sol.x - xf, LQR.P @ (ocp_sol.x - xf))
    np.testing.assert_allclose(ocp_sol.v, v_expect, atol=tol, rtol=tol)

    # Verify that interpolation of solution is close to original guess, which
    # should already be optimal
    x_int, u_int, p_int, v_int = ocp_sol(t)
    tol = np.sqrt(tol)
    np.testing.assert_allclose(x_int, x, atol=tol, rtol=tol)
    np.testing.assert_allclose(u_int, u, atol=tol, rtol=tol)
    np.testing.assert_allclose(p_int, p, atol=tol, rtol=tol)
    np.testing.assert_allclose(v_int, v, atol=tol, rtol=tol)

    # If there aren't enough nodes, the algorithm should fail
    max_nodes = ocp_sol.t.shape[0] - 10
    ocp_sol = open_loop.solve_infinite_horizon(
        ocp, t, x, u=u, p=p, method=method, t1_tol=t1_tol, max_nodes=max_nodes)

    if method == 'direct':
        assert ocp_sol.status == 10
    elif method == 'indirect':
        assert ocp_sol.status == 1
    assert ocp_sol.t.shape[0] >= max_nodes
    L = ocp.running_cost(ocp_sol.x, ocp_sol.u)
    assert L[-1] > t1_tol