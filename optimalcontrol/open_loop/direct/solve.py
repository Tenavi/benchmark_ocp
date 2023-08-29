import numpy as np

from . import utilities
from . import legendre_gauss_radau as lgr
from ._optimize import minimize
from .solutions import DirectSolution
from optimalcontrol.utilities import resize_vector


__all__ = ['solve_fixed_time', 'solve_infinite_horizon']


def solve_fixed_time(ocp, t, x, u, n_nodes=32, tol=1e-05, max_iter=500,
                     verbose=0):
    """
    ### NOT YET IMPLEMENTED

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
        `dynamics`, `jac`, and `integration_events` methods.
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
        Solution of the open-loop OCP. Should only be trusted if
        `sol.status==0`.
    """
    raise NotImplementedError


def solve_infinite_horizon(ocp, t, x, u, n_nodes=32, tol=1e-05, max_iter=500,
                           reshape_order='C', n_add_nodes=16, max_nodes=64,
                           tol_scale=1., t1_tol=1e-10, verbose=0):
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
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
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
    reshape_order : {'C', 'F'}, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering for the
        NLP decision variables. This setting can slightly affect performance.
    n_add_nodes : int, default=16
        Number of nodes to add to `n_nodes` if the running cost does not
        converge.
    max_nodes : int, default=64
        Maximum number of pseudospectral collocation nodes.
    tol_scale : float, default=1
        If the running cost does not converge and the solution is attempted
        again, multiplies `tol` by this amount. Starting with a smaller number
        of nodes and a more relaxed tolerance can help generate a good starting
        point from which the solution can be more easily improved.
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

    tol_scale = float(tol_scale)
    if tol_scale <= 0.:
        raise ValueError('tol_scale must be a positive float')
    n_add_nodes = int(n_add_nodes)
    if n_add_nodes < 1:
        raise ValueError('n_add_nodes must be a positive int')

    ocp_sol = _solve_infinite_horizon(ocp, t, x, u, tol=tol, n_nodes=n_nodes,
                                      reshape_order=reshape_order,
                                      max_iter=max_iter, verbose=verbose)

    # Stop if algorithm succeeded and running cost is small enough
    if ocp_sol.check_convergence(ocp.running_cost, t1_tol, verbose=verbose > 0):
        return ocp_sol

    # Stop if maximum nodes is reached
    if n_nodes >= max_nodes:
        ocp_sol.message = (f"n_nodes exceeded max_nodes. Last status was "
                           f"{ocp_sol.status}: {ocp_sol.message}")
        ocp_sol.status = 10
        return ocp_sol

    # Increase number of nodes if needed
    n_nodes = n_nodes + n_add_nodes
    if n_nodes > max_nodes:
        n_nodes = int(max_nodes)
    tol = tol * tol_scale

    if verbose > 0:
        print(f"Increasing n_nodes to {n_nodes:d}")

    return solve_infinite_horizon(
        ocp, t, x, u, n_nodes=n_nodes, tol=tol, max_iter=max_iter,
        reshape_order=reshape_order, n_add_nodes=n_add_nodes,
        max_nodes=max_nodes, tol_scale=tol_scale, t1_tol=t1_tol,
        verbose=verbose)


def _solve_infinite_horizon(ocp, t, x, u, n_nodes=32, tol=1e-05, max_iter=500,
                            reshape_order='C', verbose=0):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem for a single initial condition.

    This function applies a "direct method", which is to transform the optimal
    control problem into a constrained optimization problem with Legendre-Gauss-
    Radau pseudospectral collocation. The resulting optimization problem is
    solved using is solved using sequential least squares quadratic programming
    (SLSQP).

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
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
    reshape_order : {'C', 'F'}, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering for the
        NLP decision variables. This setting can slightly affect performance.
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
    # Initialize LGR quadrature
    tau, w_hat, D_hat = lgr.make_LGR(n_nodes)

    # Time scaling for transformation to LGR points
    r_tau = lgr.deriv_time_map(tau)
    w = w_hat * r_tau
    D = np.einsum('i,ij->ij', 1. / r_tau, D_hat)

    # Map initial guess to LGR points
    x0 = x[:, :1]
    x, u = utilities.interp_guess(t, x, u, tau, lgr.time_map)

    # Quadrature integration of running cost
    def cost_fun_wrapper(xu):
        x, u = utilities.separate_vars(xu, ocp.n_states, ocp.n_controls,
                                       order=reshape_order)

        L = ocp.running_cost(x, u)
        cost = np.sum(L * w)

        dLdx, dLdu = ocp.running_cost_grad(x, u, L0=L)
        jac = utilities.collect_vars(dLdx * w, dLdu * w, order=reshape_order)

        return cost, jac

    dyn_constr = utilities.make_dynamic_constraint(ocp, D, order=reshape_order)
    init_cond_constr = utilities.make_initial_condition_constraint(
        x0, ocp.n_controls, n_nodes, order=reshape_order)

    u_lb = getattr(ocp.parameters, 'u_lb', None)
    u_ub = getattr(ocp.parameters, 'u_ub', None)
    if u_lb is not None:
        u_lb = resize_vector(u_lb, ocp.n_controls)
    if u_ub is not None:
        u_ub = resize_vector(u_ub, ocp.n_controls)

    bound_constr = utilities.make_bound_constraint(u_lb, u_ub, ocp.n_states,
                                                   n_nodes, order=reshape_order)

    if verbose:
        print(f"\nNumber of LGR nodes: {n_nodes}")
        print("-" * 80)

    minimize_opts = {'maxiter': max_iter, 'iprint': verbose, 'disp': verbose}

    minimize_result = minimize(
        fun=cost_fun_wrapper,
        x0=utilities.collect_vars(x, u, order=reshape_order),
        bounds=bound_constr, constraints=[dyn_constr, init_cond_constr],
        tol=tol, jac=True, options=minimize_opts)

    return DirectSolution.from_minimize_result(
        minimize_result, ocp, tau, w, reshape_order, u_ub=u_lb, u_lb=u_ub)
