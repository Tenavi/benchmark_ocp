import warnings

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import make_interp_spline

from .solutions import OpenLoopSolution


__all__ = ['solve_fixed_time', 'solve_infinite_horizon']


class IndirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, status, message, cont_sol=None,
                 u_fun=None):
        if not callable(u_fun) or not callable(cont_sol):
            raise RuntimeError("u_fun and cont_sol must be provided at "
                               "initialization")

        self._cont_sol = cont_sol
        self._u_fun = u_fun
        self._n_states = np.shape(x)[0]
        super().__init__(t, x, u, p, v, status, message)

    def __call__(self, t, return_x=True, return_u=True, return_p=True,
                 return_v=True):
        xp = np.atleast_2d(self._cont_sol(t))
        x = xp[:self._n_states]
        p = xp[self._n_states:-1]
        v = xp[-1]

        if return_u:
            u = self._u_fun(x, p)

        return self._get_return_args(x=x if return_x else None,
                                     u=u if return_u else None,
                                     p=p if return_p else None,
                                     v=v if return_v else None)


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
        The optimal control problem to solve. Must implement `bvp_dynamics` and
        `optimal_control` methods.
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

    if u is None:
        u = ocp.optimal_control(x, p)

    if v is None:
        v = ocp.total_cost(t, x, u)[::-1]

    xp = np.vstack((x, p, np.reshape(v, (1, -1))))

    bc = _make_pontryagin_boundary(x[:, 0])

    with warnings.catch_warnings():
        np.seterr(over='warn', divide='warn', invalid='warn')
        warnings.filterwarnings('error', category=RuntimeWarning)

        try:
            bvp_sol = solve_bvp(ocp.bvp_dynamics, bc, t, xp, tol=tol,
                                max_nodes=max_nodes, verbose=verbose)

            t = bvp_sol.x
            x = bvp_sol.y[:ocp.n_states]
            p = bvp_sol.y[ocp.n_states:-1]
            v = bvp_sol.y[-1]
            u = ocp.optimal_control(x, p)

            status, message, sol = bvp_sol.status, bvp_sol.message, bvp_sol.sol
        except RuntimeWarning as w:
            status = 3
            message = str(w)
            sol = make_interp_spline(t, xp, k=1, axis=1)

    return IndirectSolution(t, x, u, p, v, status, message, cont_sol=sol,
                            u_fun=ocp.optimal_control)


def solve_infinite_horizon(ocp, t, x, p, u=None, v=None, max_nodes=1000,
                           tol=1e-05, t1_add=None, t1_max=None, t1_tol=1e-10,
                           verbose=0):
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
        The optimal control problem to solve. Must implement `bvp_dynamics` and
        `optimal_control` methods.
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
    t1_add : float, default=`t[-1]`
        Amount to increase the final time of the solution if the running cost
        does not converge on the current time horizon.
    t1_max : float, default=`t[-1] + 10 * t1_add`
        Maximum time horizon to try to integrate for before declaring failure.
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
    t1_tol = np.maximum(float(t1_tol), np.finfo(float).eps)
    _max_t = np.max(t)

    if t1_add is None:
        t1_add = _max_t
    elif np.size(t1_add) != 1 or t1_add <= 0.:
        raise ValueError("t1_add must be a positive float")

    if t1_max is None:
        t1_max = _max_t + 10. * t1_add
    elif np.size(t1_max) != 1 or t1_max <= _max_t:
        raise ValueError("t1_max must be a float greater than max(t)")

    while True:
        ocp_sol = solve_fixed_time(ocp, t, x, p, u=u, v=v, max_nodes=max_nodes,
                                   tol=tol, verbose=verbose)

        # Stop if algorithm succeeded and running cost is small enough
        if ocp_sol.check_convergence(ocp.running_cost, t1_tol, verbose=verbose):
            return ocp_sol

        # Stop if maximum nodes or time horizon is reached
        if ocp_sol.status == 1:
            return ocp_sol

        if ocp_sol.t[-1] >= t1_max:
            ocp_sol.status = 4
            ocp_sol.message = f"Maximum time horizon {t1_max} exceeded."
            return ocp_sol

        # If encountered numerical error (hard to recover from), try changing
        # the guess for the costate to all zeros
        if ocp_sol.status == 3:
            if verbose:
                print(f"Encountered numerical error. Resetting costates to zero"
                      f" and trying again...")
            ocp_sol.p = np.zeros_like(ocp_sol.p)
            ocp_sol.v = np.zeros_like(ocp_sol.v)

        # Otherwise, increase final time and continue
        t = np.concatenate([ocp_sol.t, ocp_sol.t[-1:] + t1_add])
        x = np.hstack([ocp_sol.x, ocp_sol.x[:, -1:]])
        p = np.hstack([ocp_sol.p, ocp_sol.p[:, -1:]])
        u = np.hstack([ocp_sol.u, ocp_sol.u[:, -1:]])
        v = np.concatenate([ocp_sol.v, ocp_sol.v[-1:]])

        if verbose:
            print(f"Increasing time horizon to {t[-1]}")


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
