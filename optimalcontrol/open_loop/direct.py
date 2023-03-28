import numpy as np
import warnings

from ._solve import OpenLoopSolution

try:
    import pylgr
except ImportError:
    warnings.warn(
        'Could not import pylgr library. Direct methods are not available.',
        ImportWarning)


class DirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, status, message, ps_sol=None):
        if not isinstance(ps_sol, pylgr.solve.DirectSolution):
            raise RuntimeError('ps_sol must be provided at initialization')
        self._ps_sol = ps_sol
        super().__init__(t, x, u, p, v, status, message)

    def __call__(self, t):
        x = np.atleast_2d(self._ps_sol.sol_X(t))
        u = np.atleast_2d(self._ps_sol.sol_U(t))
        p = np.atleast_2d(self._ps_sol.sol_dVdX(t))
        v = self._ps_sol.sol_V(t).reshape(-1)
        return x, u, p, v


def solve_fixed_time(ocp, t, x, u, n_nodes=32, tol=1e-05, max_iter=500,
                     verbose=0):
    """
    Compute the open-loop optimal solution of a fixed time optimal control
    problem for a single initial condition.

    This function applies a "direct method", which is to transform the optimal
    control problem into a constrained optimization problem with Legendre-Gauss-
    Lobatto pseudospectral collocation. The resulting optimization problem is
    solved using is solved using sequential least squares quadratic programming
    (SLSQP).

    Parameters
    ----------

    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    u : (n_controls, n_points) array
        Initial guess for the optimal control at times `t`.
    n_nodes : int, default=32
        Number of nodes to use in the pseudospectral discretization.
    tol : float, default=1e-05
        Convergence tolerance for the SLSQP optimizer.
    max_iter : int, default=500
        Maximum number of SLSQP iterations.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:, 0]`. Solution should only be
        trusted if `sol.status == 0`.
    """
    raise NotImplementedError('pylgr has not yet implemented finite horizon')


def solve_infinite_horizon(ocp, t, x, u, n_nodes=32, tol=1e-05, max_iter=500,
                           n_add_nodes=16, max_nodes=64, tol_scale=1.,
                           t1_tol=1e-10, verbose=0):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem for a single initial condition.

    This function applies a "direct method", which is to transform the optimal
    control problem into a constrained optimization problem with Legendre-Gauss-
    Radau pseudospectral collocation. The resulting optimization problem is
    solved using is solved using sequential least squares quadratic programming
    (SLSQP). The number of collocation nodes is increased as necessary until the
    running cost at final time `t[-1]` is smaller than the specified tolerance
    `t1_tol`.

    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    u : (n_controls, n_points) array
        Initial guess for the optimal control at times `t`.
    n_nodes : int, default=32
        Number of nodes to use in the pseudospectral discretization.
    tol : float, default=1e-05
        Convergence tolerance for the SLSQP optimizer.
    max_iter : int, default=500
        Maximum number of SLSQP iterations.
    n_add_nodes : int, default=16
        Number of nodes to add to `n_nodes` if the running cost does not
        converge.
    max_nodes : int, default=64
        Maximum number of pseudospectral collocation nodes.
    tol_scale : float, default=1
        If the running cost does not converge and the solution is attempted
        again, multiplies `tol` by this amount.
    t1_tol : float, default=1e-10
        Tolerance for the running cost when determining convergence of the
        finite horizon approximation.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:, 0]`. Solution should only be
        trusted if `sol.status == 0`.
    """
    t1_tol = float(t1_tol)

    tol_scale = float(tol_scale)
    if tol_scale <= 0.:
        raise ValueError('tol_scale must be a positive float')
    n_add_nodes = int(n_add_nodes)
    if n_add_nodes < 1:
        raise ValueError('n_add_nodes must be a positive int')

    ps_sol = pylgr.solve_ocp(ocp.dynamics, ocp.running_cost, t, x, u,
                             U_lb=getattr(ocp.parameters, 'u_lb', None),
                             U_ub=getattr(ocp.parameters, 'u_ub', None),
                             dynamics_jac=ocp.jacobians,
                             cost_grad=ocp.running_cost_gradients, tol=tol,
                             n_nodes=n_nodes, maxiter=max_iter, verbose=verbose)

    t = ps_sol.t.flatten()
    x = ps_sol.X
    u = ps_sol.U
    p = ps_sol.dVdX
    v = ps_sol.V.flatten()

    ocp_sol = DirectSolution(t, x, u, p, v, ps_sol.status, ps_sol.message,
                             ps_sol=ps_sol)

    # Stop if algorithm succeeded and running cost is small enough
    if ocp_sol.check_convergence(ocp.running_cost, t1_tol, verbose=verbose > 0):
        return ocp_sol

    # Stop if maximum nodes is reached
    if n_nodes >= max_nodes:
        ocp_sol.message = (f'n_nodes exceeded max_nodes. Last status was '
                           f'{ocp_sol.status}: {ocp_sol.message}')
        ocp_sol.status = 10
        return ocp_sol

    # Increase number of nodes if needed
    n_nodes = n_nodes + n_add_nodes
    if n_nodes > max_nodes:
        n_nodes = int(max_nodes)
    tol = tol * tol_scale

    if verbose > 0:
        print(f'Increasing n_nodes to {n_nodes:d}')

    return solve_infinite_horizon(ocp, t, x, u, n_nodes=n_nodes, tol=tol,
                                  max_iter=max_iter, n_add_nodes=n_add_nodes,
                                  max_nodes=max_nodes, tol_scale=tol_scale,
                                  t1_tol=t1_tol, verbose=verbose)
