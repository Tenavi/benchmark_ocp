import numpy as np
from scipy.integrate import solve_bvp

from .solutions import OpenLoopSolution


__all__ = ['solve_fixed_time', 'solve_infinite_horizon']


class IndirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, status, message, cont_sol=None,
                 u_fun=None):
        if not callable(cont_sol) or not callable(u_fun):
            raise RuntimeError(
                'cont_sol and u_fun must be provided at initialization')
        self._cont_sol = cont_sol
        self._u_fun = u_fun
        self._n_states = np.shape(x)[0]
        super().__init__(t, x, u, p, v, status, message)

    def __call__(self, t):
        xp = np.atleast_2d(self._cont_sol(t))
        x = xp[:self._n_states]
        p = xp[self._n_states:-1]
        v = xp[-1]
        u = self._u_fun(x, p)
        return x, u, p, v


def solve_fixed_time(ocp, t, x, p, u=None, v=None, max_nodes=1000, tol=1e-05,
                     verbose=0):
    """
    Compute the open-loop optimal solution of a fixed time optimal control
    problem for a single initial condition.

    This function applies the "indirect method", which is to solve the two-point
    boundary value problem (BVP) arising from Pontryagin's Maximum Principle
    (PMP), based on an initial guess for the optimal state and costate pair. The
    underlying BVP solver is `scipy.integrate.solve_bvp`.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    p : (n_states, n_points) array
        Initial guess for the costate at times `t`.
    u : (n_controls, n_points) array, optional
        Initial guess for the optimal control at times `t`.
    v : (n_points,) array, optional
        Initial guess for the value function at states `x`.
    max_nodes : int, default=1000
        Maximum number of collocation points to use when solving the BVP.
    tol : float, default=1e-05
        Convergence tolerance for the BVP solver.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    sol : `OpenLoopSolution`
        Solution of the open-loop OCP. Should only be trusted if
        `sol.status==0`.
    """
    t = np.reshape(t, -1)
    x = np.reshape(x, (ocp.n_states, -1))
    p = np.reshape(p, (ocp.n_states, -1))

    if v is None:
        if u is None:
            u = ocp.optimal_control(x, p)
        v = ocp.total_cost(t, x, u)[::-1]

    xp = np.vstack((x, p, np.reshape(v, (1, -1))))

    bc = _make_pontryagin_boundary(x[:, 0])

    bvp_sol = solve_bvp(ocp.bvp_dynamics, bc, t, xp, tol=tol,
                        max_nodes=max_nodes, verbose=verbose)

    t = bvp_sol.x
    x = bvp_sol.y[:ocp.n_states]
    p = bvp_sol.y[ocp.n_states:-1]
    v = bvp_sol.y[-1]
    u = ocp.optimal_control(x, p)

    return IndirectSolution(t, x, u, p, v, bvp_sol.status, bvp_sol.message,
                            cont_sol=bvp_sol.sol, u_fun=ocp.optimal_control)


def solve_infinite_horizon(ocp, t, x, p, u=None, v=None, max_nodes=1000,
                           tol=1e-05, t1_add=None, t1_tol=1e-10, verbose=0):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem for a single initial condition.

    This is accomplished by solving a series of finite horizon problems using
    `solve_finite_horizon`. The time horizons are increased in length until the
    running cost at final time `t[-1]` is smaller than the specified tolerance
    `t1_tol`.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    p : (n_states, n_points) array
        Initial guess for the costate at times `t`.
    u : (n_controls, n_points) array, optional
        Initial guess for the optimal control at times `t`.
    v : (n_points,) array, optional
        Initial guess for the value function at states `x`.
    max_nodes : int, default=1000
        Maximum number of collocation points to use when solving the BVP.
    tol : float, default=1e-05
        Convergence tolerance for the BVP solver.
    t1_add : float, optional
        Amount to increase the final time of the solution if the running cost
        does not converge on the current time horizon. If not specified, the
        default is `t1_add = t[-1]`, where `t` is the original guess for time.
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
    sol : `OpenLoopSolution`
        Solution of the open-loop OCP. Should only be trusted if
        `sol.status==0`.
    """
    t1_tol = float(t1_tol)

    if t1_add is None:
        t1_add = np.max(t)
    elif np.size(t1_add) != 1 or t1_add <= 0.:
        raise ValueError('t1_add must be a positive float')

    while True:
        ocp_sol = solve_fixed_time(ocp, t, x, p, u=u, v=v, max_nodes=max_nodes,
                                   tol=tol, verbose=verbose)

        # Stop if maximum nodes is reached
        if ocp_sol.status == 1:
            return ocp_sol

        # Stop if algorithm succeeded and running cost is small enough
        if ocp_sol.check_convergence(ocp.running_cost, t1_tol,
                                     verbose=verbose > 0):
            return ocp_sol

        # Otherwise, increase final time and continue
        t = np.concatenate([ocp_sol.t, ocp_sol.t[-1:] + t1_add])
        x = np.hstack([ocp_sol.x, ocp_sol.x[:, -1:]])
        p = np.hstack([ocp_sol.p, np.zeros((ocp.n_states, 1))])
        u = np.hstack([ocp_sol.u, ocp_sol.u[:, -1:]])
        v = np.concatenate([ocp_sol.v, ocp_sol.v[-1:]])

        if verbose > 0:
            print(f'\nIncreasing time horizon to {t[-1]}')


def _make_pontryagin_boundary(x0):
    """
    Generates a function to evaluate the boundary conditions for a given initial
    condition. Terminal cost is zero so final condition on costate and value
    function are both zero.

    Parameters
    ----------
    x0 : (n_states,) array
        Initial condition.

    Returns
    -------
    bc : callable
        Function of `xp_0` (augmented states at initial time) and `xp_1`
        (augmented states at final time), returning a function which evaluates
        to zero if the boundary conditions are satisfied.
    """
    x0 = np.reshape(x0, -1)
    n_states = x0.shape[0]

    def bc(xp_0, xp_1):
        return np.concatenate((xp_0[:n_states] - x0, xp_1[n_states:]))

    return bc
