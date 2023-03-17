import pytest
import numpy as np

from optimalcontrol.open_loop import indirect
from optimalcontrol.simulate import integrate_closed_loop
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params


@pytest.mark.parametrize("method", ["indirect"])
def test_solve_infinite_horizon(method):
    """
    Basic test of an LQR-controlled linear system integrated over a fixed time
    horizon. Compares BVP results with LQR, which is known to be optimal.
    """
    n_states = 3
    n_controls = 2
    t1_tol = 1e-14

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=123)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, x0_lb=-1.,
                                 x0_ub=1., u_lb=-1., u_ub=1.,
                                 x0_sample_seed=456)
    LQR = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, u_lb=-1.,
                                   u_ub=1.)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    # Integrate over initially short time horizon
    t_span = [0., 10.]
    t, x, _ = integrate_closed_loop(ocp, LQR, t_span, x0)
    p = 2. * LQR.P @ (x - xf)
    u = LQR(x)
    v = np.einsum("ij,ij->j", x - xf, LQR.P @ (x - xf))

    ocp_sol, status = indirect.solve_infinite_horizon(
        ocp, t, x, p, max_nodes=5000, t1_tol=t1_tol)

    assert status == 0
    assert ocp_sol.t[-1] > t_span[-1]
    L = ocp.running_cost(ocp_sol.x, ocp_sol.u)
    assert L[-1] <= t1_tol

    # Verify that BVP solution matches LQR solution, which is optimal
    u_expect = LQR(ocp_sol.x)
    np.testing.assert_allclose(ocp_sol.u, u_expect, atol=1e-06, rtol=1e-06)
    p_expect = 2. * LQR.P @ (ocp_sol.x - xf)
    np.testing.assert_allclose(ocp_sol.p, p_expect, atol=1e-06, rtol=1e-06)
    v_expect = np.einsum("ij,ij->j", ocp_sol.x - xf, LQR.P @ (ocp_sol.x - xf))
    np.testing.assert_allclose(ocp_sol.v, v_expect, atol=1e-06, rtol=1e-06)

    # Verify that interpolation of solution is close to original guess, which
    # should already be optimal
    x_int, u_int, p_int, v_int = ocp_sol(t)
    np.testing.assert_allclose(x_int, x, atol=1e-03, rtol=1e-03)
    np.testing.assert_allclose(u_int, u, atol=1e-03, rtol=1e-03)
    np.testing.assert_allclose(p_int, p, atol=1e-03, rtol=1e-03)
    np.testing.assert_allclose(v_int, v, atol=1e-03, rtol=1e-03)

    # If there aren't enough nodes, the algorithm should fail
    max_nodes = ocp_sol.t.shape[0] - 10
    ocp_sol, status = indirect.solve_infinite_horizon(
        ocp, t, x, p, max_nodes=max_nodes, t1_tol=t1_tol)

    assert status == 1
    assert ocp_sol.t.shape[0] >= max_nodes
    L = ocp.running_cost(ocp_sol.x, ocp_sol.u)
    assert L[-1] > t1_tol
