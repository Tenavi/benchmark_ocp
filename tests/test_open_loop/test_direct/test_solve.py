import pytest
import time

from optimalcontrol.open_loop import direct

from tests.test_open_loop.test_indirect import (setup_lqr_test, get_lqr_sol,
                                                assert_matches_reference)


@pytest.mark.parametrize('u_bound', (None, .75))
@pytest.mark.parametrize('order', ('C', 'F'))
@pytest.mark.parametrize('n_nodes', (36, 37))
def test_single_solve_infinite_horizon_lqr(u_bound, order, n_nodes):
    """
    Basic test of an LQR-controlled linear system. The OCP is solved over an
    approximate infinite horizon and compared with LQR, which is known to be
    optimal.
    """
    n_states = 3
    n_controls = 2

    t1_init = 10.

    kwargs = {'n_nodes': n_nodes, 'reshape_order': order}

    atol = 0.05
    rtol = 0.05

    ocp, lqr = setup_lqr_test(n_states, n_controls, u_bound=u_bound, seed=123)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1.)
    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init)

    start_time = time.time()

    ocp_sol = direct.solve._solve_infinite_horizon(ocp, t, x, u, **kwargs)

    comp_time = time.time() - start_time
    print(f'Solution time: {comp_time:1.2f} sec')

    assert ocp_sol.status == 0

    assert_matches_reference(ocp_sol, t, x, u, atol=atol, rtol=rtol)

    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init, t_eval=ocp_sol.t)

    assert_matches_reference(ocp_sol, t, x, u, p=p if u_bound is None else None,
                             atol=atol, rtol=rtol)


@pytest.mark.parametrize('u_bound', (None, .75))
@pytest.mark.parametrize('order', ('C', 'F'))
@pytest.mark.parametrize('n_nodes', (11, 12))
def test_solve_infinite_horizon_lqr(u_bound, order, n_nodes):
    """
    Basic test of an LQR-controlled linear system. The OCP is solved over an
    approximate infinite horizon and compared with LQR, which is known to be
    optimal.
    """
    n_states = 3
    n_controls = 2

    t1_init = 10.

    t1_tol = 1e-12

    kwargs = {'n_nodes': n_nodes, 'reshape_order': order,
              'max_nodes': 64, 't1_tol': t1_tol}

    atol = 0.05
    rtol = 0.05

    ocp, lqr = setup_lqr_test(n_states, n_controls, u_bound=u_bound, seed=123)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1.)
    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init)

    start_time = time.time()

    ocp_sol = direct.solve.solve_infinite_horizon(ocp, t, x, u, **kwargs)

    comp_time = time.time() - start_time
    print(f'Solution time: {comp_time:1.2f} sec')

    assert ocp_sol.status == 0
    assert n_nodes < ocp_sol.t.size <= kwargs['max_nodes']
    assert ocp.running_cost(ocp_sol.x, ocp_sol.u)[-1] <= t1_tol

    assert_matches_reference(ocp_sol, t, x, u, atol=atol, rtol=rtol)

    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init, t_eval=ocp_sol.t)

    assert_matches_reference(ocp_sol, t, x, u, p=p if u_bound is None else None,
                             atol=atol, rtol=rtol)
