import numpy as np

from . import setup_nlp, radau
from ._optimize import minimize
from .solutions import DirectSolution
from optimalcontrol.open_loop.solutions import CombinedSolution
from optimalcontrol.simulate._ivp import solve_ivp


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
        The optimal control problem to solve.
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
                           t1_tol=1e-06, interp_tol=1e-02, max_n_segments=10,
                           integration_method='RK45', atol=1e-08, rtol=1e-04,
                           reshape_order='F', verbose=0):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem for a single initial condition.

    This function applies a "direct method", which is to transform the optimal
    control problem into a constrained optimization problem with Legendre-Gauss-
    Radau pseudospectral collocation. The resulting optimization problem is
    solved using is solved using sequential least squares quadratic programming
    (SLSQP).

    The solution is "antialiased" by stringing multiple solutions together. The
    individual solutions are referred to as "Bellman segments". Each of these
    smaller subproblems is meant to be easier to solve than a single problem
    over a longer time horizon and using a much larger number of collocation
    nodes. This is referred to as the "a2B" algorithm following ref. [1].

    The start of a new segment is determined by integrating the system with the
    open-loop control, either until the running cost is smaller than the
    specified tolerance `t1_tol`, at which point convergence is declared, or the
    interpolation error between the integrated state and the open-loop optimal
    state is larger than specified tolerance `interp_tol`. This interpolation
    error is calculated as
    ```
    scale_factor = max([norm(x), norm(x_interp), 1])
    error = norm(x - x_interp) / scale_factor
    ```
    where `x` is the integrated state and `x_interp` is the open-loop optimal
    state trajectory.

    ##### References

    1. I. M. Ross, Q. Gong, and P. Sekhavat, *Low-thrust, high-accuracy
        trajectory optimization*, Journal of Guidance, Control, and Dynamics, 30
        (2007), pp. 921-933. https://doi.org/10.2514/1.23181

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        The optimal control problem to solve.
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
    t1_tol : float, default=1e-06
        Tolerance for the running cost when determining convergence of the
        finite horizon approximation.
    interp_tol : float, default=1e-02
        Tolerance for the relative interpolation error (see above) when
        determining when to start a new Bellman segment.
    max_n_segments : int, default=10
        The maximum number of Bellman segments to connect to form the complete
        solution.
    integration_method : string or `OdeSolver`, default='RK45'
        The solver to use for integration in the a2B algorithm. See
        `scipy.integrate.solve_ivp` for options.
    atol : float or array_like, default=1e-08
        Absolute tolerance for integration in the a2B algorithm. See
        `scipy.integrate.solve_ivp`.
    rtol : float or array_like, default=1e-04
        Relative tolerance for integration in the a2B algorithm. See
        `scipy.integrate.solve_ivp`.
    reshape_order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering for the NLP
        decision variables. This setting can slightly affect performance.
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
    interp_tol = np.maximum(float(interp_tol), np.finfo(float).eps)
    max_n_segments = np.maximum(int(max_n_segments), 1)

    f, converge_event, interp_event = _setup_open_loop(ocp, t1_tol, interp_tol)

    sols, t_break = [], []

    for _ in range(max_n_segments):
        sols.append(_solve_infinite_horizon(ocp, t, x, u, tol=tol,
                                            n_nodes=n_nodes, max_iter=max_iter,
                                            reshape_order=reshape_order,
                                            verbose=verbose))

        ode_sol = solve_ivp(f, [0., sols[-1].t[-1]], sols[-1].x[:, 0],
                            events=(converge_event, interp_event),
                            args=(sols[-1],), exact_event_times=True,
                            method=integration_method, atol=atol, rtol=rtol,
                            vectorized=True)

        # Integration failed
        if ode_sol.status == -1:
            sols[-1].status = -1
            sols[-1].message = ode_sol.message
            break

        # Integration reached the end of the time horizon or the running cost
        # converged
        if ode_sol.status == 0 or ode_sol.t_events[0].size >= 1:
            break

        if len(sols) >= max_n_segments:
            sols[-1].status = 4
            sols[-1].message = (f"Reached maximum number of Bellman "
                                f"segments ({max_n_segments})")
            break

        # Otherwise, the interpolation error is greater than the tolerance, so
        # solve a new OCP
        t1 = ode_sol.t[-1]

        if len(t_break) >= 1:
            t_break.append(t1 + t_break[-1])
        else:
            t_break.append(t1)

        # Get the initial guess for the next segment
        idx = sols[-1].t >= t1

        # Time guess starts at zero
        # In case t[0] == t1, add a small offset so time points are different
        t = sols[-1].t[idx] - t1 + 1e-07
        t = np.concatenate(([0.], t))

        # Set the initial condition to the one obtained by integration
        x, u = sols[-1](t, return_p=False, return_v=False)
        x[:, 0] = ode_sol.y[:, -1]

        if verbose:
            print(f"Starting new Bellman segment at t1 = {t_break[-1]}")
            print(f"Running cost L(t1) = {ocp.running_cost(x[:, 0], u[:, 0])}")

    if len(sols) == 1:
        return sols[0]

    return CombinedSolution(sols, t_break)


def _solve_infinite_horizon(ocp, t, x, u, n_nodes=16, tol=1e-05, max_iter=500,
                            reshape_order='F', verbose=0):
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
        The optimal control problem to solve.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    u : (n_controls, n_points) array
        Initial guess for the optimal control at times `t`.
    n_nodes : int, default=16
        Number of nodes to use in the pseudospectral discretization.
    tol : float, default=1e-05
        Convergence tolerance for the SLSQP optimizer.
    max_iter : int, default=500
        Maximum number of SLSQP iterations.
    reshape_order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering for the NLP
        decision variables. This setting can slightly affect performance.
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
    tau, w, D = radau.make_scaled_lgr(n_nodes)
    cost_fun, dyn_constr, bound_constr = setup_nlp.setup(
        ocp, tau, w, D, order=reshape_order)

    # Map initial guess to LGR points
    x, u = setup_nlp.interp_guess(t, x, u, tau, radau.inverse_time_map)
    xu = setup_nlp.collect_vars(x, u, order=reshape_order)

    x0_constr = setup_nlp.make_initial_condition_constraint(
        x[:, :1], ocp.n_controls, n_nodes, order=reshape_order)

    if verbose:
        print(f"\nNumber of LGR nodes: {n_nodes}")
        print("-" * 80)

    minimize_opts = {'maxiter': max_iter, 'iprint': verbose, 'disp': verbose}

    minimize_result = minimize(cost_fun, xu, bounds=bound_constr,
                               constraints=[dyn_constr, x0_constr],
                               tol=tol, jac=True, options=minimize_opts)

    return DirectSolution.from_minimize_result(minimize_result, ocp, tau, w,
                                               order=reshape_order)


def _setup_open_loop(ocp, t1_tol, interp_tol):
    # Open-loop controlled system dynamics
    def dynamics(t, x, sol):
        u_interp = sol(t, return_x=False, return_p=False, return_v=False)
        if x.ndim < 2:
            u_interp = u_interp.reshape(-1, )
        return ocp.dynamics(x, u_interp)

    # Terminate integration for sufficiently small running cost
    def running_cost_converged(t, x, sol):
        u_interp = sol(t, return_x=False, return_p=False, return_v=False)
        if x.ndim < 2:
            u_interp = u_interp.reshape(-1, )
        return ocp.running_cost(x, u_interp) - t1_tol

    # Terminate integration for exceeding interpolation error tolerance
    def error_exceeds_interp_tol(t, x, sol):
        x_interp = sol(t, return_u=False, return_p=False, return_v=False)
        x_interp = x_interp.reshape(x.shape)

        # Get denominator for relative norm
        x_norm = np.linalg.norm(x, axis=0)
        x_interp_norm = np.linalg.norm(x_interp, axis=0)
        scale_factor = np.maximum(np.maximum(x_norm, x_interp_norm), 1)

        return np.linalg.norm(x - x_interp, axis=0) / scale_factor - interp_tol

    # Assume running cost event starts positive, terminate when becomes negative
    running_cost_converged.terminal = True
    running_cost_converged.direction = -1

    # Assume error event starts negative, terminate when becomes positive
    error_exceeds_interp_tol.terminal = True
    error_exceeds_interp_tol.direction = 1

    return dynamics, running_cost_converged, error_exceeds_interp_tol
