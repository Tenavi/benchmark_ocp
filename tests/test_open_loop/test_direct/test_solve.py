import time

import numpy as np
import pytest

from optimalcontrol.open_loop import direct
from optimalcontrol.simulate._ivp import solve_ivp

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

    # Check that the solution is correct at collocation nodes
    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init, t_eval=ocp_sol.t)

    assert_matches_reference(ocp_sol, t, x, u, p=p if u_bound is None else None,
                             atol=atol, rtol=rtol)


@pytest.mark.parametrize('t1_tol', ('easy', 'strict'))
def test_open_loop_dynamics_and_events(t1_tol):
    n_states = 3
    n_controls = 2
    u_bound = 0.75

    t1 = 10.

    interp_tol = 0.1

    ocp, lqr = setup_lqr_test(n_states, n_controls, u_bound=u_bound, seed=123)
    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1.)
    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1)

    ocp_sol = direct.solve._solve_infinite_horizon(ocp, t, x, u)

    # To make an easy convergence tolerance, set the limit to be the optimal
    # running cost close to the start of the trajectory
    L0 = ocp.running_cost(x0, u[:, 0])
    if t1_tol == 'easy':
        t1_tol_num = L0 * 1e-01
    elif t1_tol == 'strict':
        t1_tol_num = L0 * 1e-04
    else:
        raise NotImplementedError

    f, converge_event, interp_event = direct.solve._setup_open_loop(
        ocp, t1_tol_num, interp_tol)

    assert converge_event.terminal
    assert converge_event.direction < 0
    assert interp_event.terminal
    assert interp_event.direction > 0

    # Make sure we start in the correct place
    assert converge_event(0., x0, ocp_sol) > 0.
    assert interp_event(0., x0, ocp_sol) < 0.

    ode_sol = solve_ivp(f, [0., ocp_sol.t[-1]], x0, exact_event_times=True,
                        events=(converge_event, interp_event), args=(ocp_sol,))

    converge_event = converge_event(ode_sol.t, ode_sol.y, ocp_sol)
    interp_event = interp_event(ode_sol.t, ode_sol.y, ocp_sol)

    if t1_tol == 'easy':
        # The solution's running cost should converge before interpolation error
        # takes over
        assert ode_sol.t_events[0].size == 1
        np.testing.assert_allclose(converge_event[-1], 0., atol=1e-12)
        assert interp_event[-1] < 0.
    else:
        # The interpolation error should be unacceptably large before the
        # running cost is small enough
        assert ode_sol.t_events[1].size == 1
        np.testing.assert_allclose(interp_event[-1], 0., atol=1e-12)
        assert converge_event[-1] > 0.


@pytest.mark.parametrize('u_bound', (None, .75))
@pytest.mark.parametrize('order', ('C', 'F'))
@pytest.mark.parametrize('n_nodes', (11, 12))
def test_solve_infinite_horizon_lqr_old(u_bound, order, n_nodes):
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

    ocp_sol = direct.solve._solve_infinite_horizon_old(ocp, t, x, u, **kwargs)

    comp_time = time.time() - start_time
    print(f'Solution time: {comp_time:1.2f} sec')

    assert ocp_sol.status == 0
    assert n_nodes < ocp_sol.t.size <= kwargs['max_nodes']
    assert ocp.running_cost(ocp_sol.x, ocp_sol.u)[-1] <= t1_tol

    assert_matches_reference(ocp_sol, t, x, u, atol=atol, rtol=rtol)

    t, x, p, u = get_lqr_sol(ocp, lqr, x0, t1_init, t_eval=ocp_sol.t)

    print(f"n_nodes: {ocp_sol.t.size}, value: {ocp_sol.v[0]:.4f}, max error: {np.abs(ocp_sol(t)[1] - u).max():.4f}")

    assert_matches_reference(ocp_sol, t, x, u, p=p if u_bound is None else None,
                             atol=atol, rtol=rtol)
