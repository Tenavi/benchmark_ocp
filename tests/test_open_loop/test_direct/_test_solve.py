import pytest

import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

from optimalcontrol.open_loop.direct.ps_solution import solve_ocp

from .test_data import example_problems

PLOT_SIMS = False

def _assert_converged(PS_sol, tol):
    assert PS_sol.success
    assert PS_sol.residuals.max() < tol

def _get_LQR_guess(OCP, t1, X0, tol):
    def closed_loop_dynamics(t, X):
        U = OCP.LQR_control(X)
        return OCP.dynamics(X, U)

    LQR_sol = solve_ivp(
        closed_loop_dynamics, [0.,t1], X0.flatten(), atol=tol/1000, rtol=tol
    )

    t_LQR = LQR_sol.t
    X_LQR = LQR_sol.y
    U_LQR = OCP.LQR_control(X_LQR)

    LQR_cost = OCP.running_cost(X_LQR, U_LQR)
    LQR_cost = np.trapz(LQR_cost, x=t_LQR)

    return t_LQR, X_LQR, U_LQR, LQR_cost

def _get_BVP_sol(OCP, t_guess, X_guess, tol):
    X0 = X_guess[:,:1]
    X_aug_guess = np.vstack((X_guess, 2.*np.matmul(OCP.P, X_guess)))
    BVP_sol = solve_bvp(
        OCP.bvp_dynamics, OCP.make_bc(X0), t_guess, X_aug_guess, tol=tol
    )

    t_opt = BVP_sol.x
    X_opt = BVP_sol.y[:OCP.n_states]
    dVdX_opt = BVP_sol.y[OCP.n_states:]
    U_opt = OCP.U_star(X_opt, BVP_sol.y[OCP.n_states:])

    opt_cost = OCP.running_cost(X_opt, U_opt)
    opt_cost = np.trapz(opt_cost, x=t_opt)

    return t_opt, X_opt, U_opt, dVdX_opt, opt_cost

@pytest.mark.parametrize('U_max', [None,.25])
@pytest.mark.parametrize('order', ['C','F'])
@pytest.mark.parametrize('n_nodes', [11,16])
def test_LQR(U_max, order, n_nodes):
    '''
    Evaluate the solve_ocp method against a reference LQR solution with and
    without control saturation constraints.
    '''
    tol = 1e-05
    plot_sims = PLOT_SIMS
    verbose = 0
    n_x, n_u, n_t = 3, 2, n_nodes

    OCP = example_problems.LinearSystem(n_x, n_u, U_max, seed=1234)

    X0 = np.array([[-0.0085], [1.6218], [1.1564]])
    t1 = 30.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, dVdX_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/10.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub,
        dynamics_jac=OCP.jacobians, cost_grad=OCP.running_cost_gradient,
        n_nodes=n_nodes, tol=tol, maxiter=10000,
        reshape_order=order, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V.flatten()[0])
    )

    _assert_converged(PS_sol, tol)
    assert PS_sol.V.flatten()[0] < opt_cost * 1.1

@pytest.mark.parametrize('order', ['C'])
@pytest.mark.parametrize('n_nodes', [11,32])
def test_van_der_pol(order, n_nodes):
    '''
    Evaluate the solve_ocp method against a reference solution obtained with an
    indirect method.
    '''
    tol = 1e-05
    plot_sims = PLOT_SIMS
    verbose = 0

    OCP = example_problems.VanDerPol()

    X0 = np.array([[-2.3997], [3.2243]])
    t1 = 30.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, dVdX_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/10.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub,
        dynamics_jac=OCP.jacobians, cost_grad=OCP.running_cost_gradient,
        n_nodes=n_nodes, tol=tol, maxiter=10000,
        reshape_order=order, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V.flatten()[0])
    )

    _assert_converged(PS_sol, tol)
    if n_nodes >= 20:
        assert PS_sol.V.flatten()[0] < opt_cost * 1.1

@pytest.mark.parametrize('order', ['C'])
@pytest.mark.parametrize('n_nodes', [11,32,40])
def test_satellite(order, n_nodes):
    '''
    Evaluate the solve_ocp method against a reference solution obtained with an
    indirect method.
    '''
    tol = 1e-05
    plot_sims = PLOT_SIMS
    verbose = 0

    OCP = example_problems.Satellite()

    X0 = np.array([
        [0.17678],
        [0.91856],
        [0.17678],
        [0.30619],
        [0.],
        [0.],
        [0.]
    ])
    t1 = 120.

    start_time = time.time()

    t_LQR, X_LQR, U_LQR, LQR_cost = _get_LQR_guess(OCP, t1, X0, tol)
    t_opt, X_opt, U_opt, dVdX_opt, opt_cost = _get_BVP_sol(OCP, t_LQR, X_LQR, tol/100.)

    print(
        '\nIndirect solution time: %.4fs. Optimal cost: %.4f'
        % (time.time() - start_time, opt_cost)
    )

    start_time = time.time()

    PS_sol = solve_ocp(
        OCP.dynamics, OCP.running_cost, t_LQR, X_LQR, U_LQR,
        U_lb=OCP.U_lb, U_ub=OCP.U_ub,
        dynamics_jac=OCP.jacobians, cost_grad=OCP.running_cost_gradient,
        n_nodes=n_nodes, tol=tol, maxiter=10000,
        reshape_order=order, verbose=verbose
    )

    print(
        'Direct solution time:   %.4fs. LGR PS cost:  %.4f'
        % (time.time() - start_time, PS_sol.V.flatten()[0])
    )

    _assert_converged(PS_sol, tol)
    if n_nodes >= 40:
        assert PS_sol.V.flatten()[0] < opt_cost * 1.1
