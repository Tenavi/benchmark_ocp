import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline

from ._solve import OpenLoopSolution


class IndirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, bvp_sol=None, u_fun=None):
        if not callable(bvp_sol) or not callable(u_fun):
            raise RuntimeError(
                "bvp_sol and u_fun must be provided at initialization"
            )
        self._bvp_sol = bvp_sol
        self._u_fun = u_fun
        self._v_fun = CubicSpline(t, np.reshape(v, (-1,)))
        self._n_states = np.shape(x)[0]
        super().__init__(t, x, u, p, v)

    def __call__(self, t):
        xp = np.atleast_2d(self._bvp_sol(t))
        x = xp[:self._n_states]
        p = xp[self._n_states:]
        u = self._u_fun(x, p)
        v = self._v_fun(t)
        return x, u, p, v


def solve_fixed_time(ocp, t, x, p, u=None, v=None, max_nodes=1000, tol=1e-05,
                     verbose=0):
    """
    Compute the open-loop optimal solution for a single initial condition
    given an initial guess.


    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : array_like, shape (n_points,)
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : array_like, shape (n_states, n_points)
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    p : array_like, shape (n_states, n_points)
        Initial guess for the costate at times `t`.
    u : array_like, shape (n_controls, n_points), optional
        Initial guess for the optimal control at times `t`.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:,0]`.
    success : bool
        `True` if the algorithm succeeded.
    """
    t = np.reshape(t, -1)
    x = np.reshape(x, (ocp.n_states, -1))
    p = np.reshape(p, (ocp.n_states, -1))
    xp = np.vstack((x, p))

    bc = make_pontryagin_boundary(x[:, 0])

    bvp_sol = solve_bvp(ocp.bvp_dynamics, bc, t, xp, tol=tol,
                        max_nodes=max_nodes, verbose=verbose)

    t = bvp_sol.x
    x = bvp_sol.y[:ocp.n_states]
    p = bvp_sol.y[ocp.n_states:]
    u = ocp.optimal_control(x, p)
    self.sol['V'] = self.bvp_sol.y[-1:]
    self.sol['U'] = self.ocp.U_star(self.sol['X'], self.sol['dVdX'])


def solve_infinite_horizon(ocp, t, x, p, max_nodes=1000, tol=1e-05,
                           t1_scale=3/2, verbose=0):
    """
    Compute the open-loop optimal solution for a single initial condition
    given an initial guess.


    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : array_like, shape (n_points,)
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : array_like, shape (n_states, n_points)
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    p : array_like, shape (n_states, n_points)
        Initial guess for the costate at times `t`.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:,0]`.
    success : bool
        `True` if the algorithm succeeded.
    """
    raise NotImplementedError


def _extend_horizon(self):
    if self.bvp_sol is None:
        return False
    # Cannot extend horizon if exceeded number of mesh nodes or maximum time
    if self.bvp_sol.status == 1 or self.sol['t'][-1] >= self.t1_max:
        return False

    self.sol['t'][-1] = np.minimum(
        self.t1_max, self.sol['t'][-1]*self.t1_scale
    )

    return True


def make_pontryagin_boundary(x0):
    """
    Generates a function to evaluate the boundary conditions for a given initial
    condition. Terminal cost is zero so final condition on costate is zero.

    Parameters
    ----------
    x0 : array_like, shape (n_states,)
        Initial condition.

    Returns
    -------
    bc : callable
        Function of xp_0 (augmented states at initial time) and xp_1 (augmented
        states at final time), returning a function which evaluates to zero if
        the boundary conditions are satisfied.
    """
    x0 = np.reshape(x0, -1)
    n_states = x0.shape[0]

    def bc(xp_0, xp_1):
        return np.concatenate((xp_0[:n_states] - x0, xp_1[n_states:]))

    return bc
