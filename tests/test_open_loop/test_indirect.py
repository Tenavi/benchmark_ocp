import pytest
import numpy as np
import time

from optimalcontrol.open_loop import indirect
from optimalcontrol.simulate import integrate_fixed_time
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from tests._utilities import make_LQ_params


def setup_lqr_test(n_states, n_controls, u_bound=None, seed=None):
    if u_bound is None:
        u_lb, u_ub = None, None
    else:
        u_lb, u_ub = -u_bound, u_bound

    if seed is None:
        seed = int(time.time())

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls, seed=seed)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                 x0_lb=-1., x0_ub=1., u_lb=u_lb, u_ub=u_ub,
                                 x0_sample_seed=seed + 1)
    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                                   u_lb=u_lb, u_ub=u_ub)
    
    return ocp, lqr
    

def get_lqr_sol(ocp, lqr, x0, t1, t_eval=None):
    if t_eval is None:
        t_span = [0., t1]
    else:
        t_span = [t_eval[0], t_eval[-1]]

    t, x, _ = integrate_fixed_time(ocp, lqr, x0, t_span, t_eval=t_eval)
    p = 2. * lqr.P @ (x - lqr.xf)
    u = lqr(x)

    return t, x, p, u


def assert_matches_reference(ocp_sol, t, x, u, p=None, atol=1e-02, rtol=1e-02):
    x_int, u_int, p_int, v_int = ocp_sol(t)

    np.testing.assert_allclose(x_int, x, atol=atol, rtol=rtol)
    np.testing.assert_allclose(u_int, u, atol=atol, rtol=rtol)
    if p is not None:
        np.testing.assert_allclose(p_int, p, atol=atol, rtol=rtol)


@pytest.mark.parametrize('u_bound', (None, 0.75))
def test_solve_infinite_horizon_lqr(u_bound):
    """
    Basic test of an LQR-controlled linear system. The OCP is solved over an
    approximate infinite horizon and compared with LQR, which is known to be
    optimal. This test implicitly tests `indirect.solve_fixed_time`, since this
    is used for `indirect.solve_infinite_horizon`.
    """
    n_states = 3
    n_controls = 2
    
    t1_init = 10.

    t1_tol = 1e-12

    atol = 0.005
    rtol = 0.005

    ocp, lqr = setup_lqr_test(n_states, n_controls, u_bound=u_bound, seed=123)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1.)
    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init)

    start_time = time.time()

    ocp_sol = indirect.solve_infinite_horizon(ocp, t, x, p=p, t1_tol=t1_tol,
                                              max_nodes=1000)

    comp_time = time.time() - start_time
    print(f'Solution time: {comp_time:1.2f} sec')

    assert ocp_sol.status == 0
    assert ocp_sol.t[-1] > t1_init
    assert ocp.running_cost(ocp_sol.x, ocp_sol.u)[-1] <= t1_tol

    assert_matches_reference(ocp_sol, t, x, u, atol=atol, rtol=rtol)

    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init, t_eval=ocp_sol.t)

    assert_matches_reference(ocp_sol, t, x, u, p=p if u_bound is None else None,
                             atol=atol, rtol=rtol)
