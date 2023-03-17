import numpy as np
import warnings

from ._solve import OpenLoopSolution

try:
    import pylgr
except:
    warnings.warn(
        "Could not import pylgr library. DirectSolver is not available.",
        ImportWarning
    )


class DirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, dVdx, V, ps_sol=None):
        if not isinstance(ps_sol, pylgr.solve.DirectSolution):
            raise TypeError("ps_sol must be provided at initialization")
        self._ps_sol = ps_sol
        super().__init__(t, x, u, dVdx, V)

    def __call__(self, t):
        x = np.atleast_2d(self._ps_sol.sol_X(t)
        u = np.atleast_2d(self.ps_sol.sol_U(t))
        dVdx = np.atleast_2d(self.ps_sol.sol_dVdX(t))
        V = self.ps_sol.sol_V(t)
        return x, u, dVdx, V


def solve_fixed_time(ocp, t, x, u, n_nodes=16, tol=1e-05, max_iter=500,
                     verbose=0):
    """
    Compute the open-loop optimal solution for a single initial condition
    given an initial guess.


    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : ndarray, shape (n_points,)
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : ndarray, shape (n_states, n_points)
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    u : ndarray, shape (n_controls, n_points)
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
    raise NotImplementedError

    self.ps_sol = pylgr.solve_ocp(
        self.OCP.dynamics, self.OCP.running_cost, t, X, U,
        U_lb=self.OCP.U_lb, U_ub=self.OCP.U_ub,
        dynamics_jac=self.OCP.jacobians,
        cost_grad=self.OCP.running_cost_gradient,
        tol=self.tol, n_nodes=self.n_nodes, maxiter=self.max_iter,
        verbose=verbose
    )

    self.sol['t'] = self.ps_sol.t.flatten()
    self.sol['X'] = self.ps_sol.X
    self.sol['dVdX'] = self.ps_sol.dVdX
    self.sol['V'] = self.ps_sol.V
    self.sol['U'] = self.ps_sol.U


def solve_infinite_horizon(ocp, t, x, u, n_nodes=16, tol=1e-05, max_iter=500,
                           n_add_nodes=16, max_nodes=64, tol_scale=1.,
                           verbose=0):
    """
    Compute the open-loop optimal solution for a single initial condition
    given an initial guess.


    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    t : ndarray, shape (n_points,)
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : ndarray, shape (n_states, n_points)
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    u : ndarray, shape (n_controls, n_points)
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
    raise NotImplementedError


def _extend_horizon(self):
    if self.ps_sol is None:
        return False
    # Cannot extend horizon if exceeded number of mesh nodes
    if self.n_nodes >= self.max_nodes:
        return False

    self.n_nodes = min(self.max_nodes, self.n_nodes+self.n_add_nodes)
    self.tol = self.tol * self.tol_scale

    return True


