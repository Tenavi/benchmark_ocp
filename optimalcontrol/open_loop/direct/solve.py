import numpy as np

from . import setup_nlp, radau, time_maps
from ._optimize import minimize
from .solutions import DirectSolution
from optimalcontrol.open_loop.solutions import CombinedSolution
from optimalcontrol.simulate._ivp import solve_ivp
from optimalcontrol.simulate.simulate import _make_state_bound_events


__all__ = ['solve_fixed_time', 'solve_infinite_horizon']


def solve_fixed_time(ocp, t, x, u, n_nodes=32, n_nodes_init=None, tol=1e-06,
                     max_iter=500, reshape_order='F', verbose=0):
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
    tol : float, default=1e-06
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


def solve_infinite_horizon(ocp, t, x, u, time_map=time_maps.TimeMapLog2,
                           n_nodes=64, n_nodes_init=None, tol=1e-06,
                           max_iter=500, t1_tol=1e-06, interp_tol=1e-03,
                           max_n_segments=10, integration_method='RK45',
                           atol=1e-08, rtol=1e-04, reshape_order='F',
                           verbose=0):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem for a single initial condition.

    This function applies a "direct method", which is to transform the optimal
    control problem into a constrained optimization problem with Legendre-Gauss-
    Radau pseudospectral collocation. The resulting optimization problem is
    solved using is solved using sequential least squares quadratic programming
    (SLSQP). Each optimization is performed twice: first with fewer collocation
    points (`n_nodes_init`) and then more points (`n_nodes`) to refine the
    solutions.

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
    2. D. Garg, W. W. Hager, and A. V. Rao, *Pseudospectral methods for solving
        infinite-horizon optimal control problems*, Automatica, 47 (2011), pp.
        829-837. https://doi.org/10.1016/j.automatica.2011.01.085

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
    time_map : `TimeMapRadau`, default=`TimeMapLog2`
        `TimeMapRadau` subclass implementing `physical_to_radau`,
        `radau_to_physical`, and `derivative` methods. The default maps
        `radau_to_physical(tau) = log(4 / (1 - tau) ** 2)` as in eq (4) from
        ref. [2] above.
    n_nodes : int, default=64
        Number of nodes to use in the pseudospectral discretization for the
        final solution. `n_nodes` must be at least 4.
    n_nodes_init : array of ints, default=`n_nodes // 2`
        Number of nodes to use in the pseudospectral discretization of the
        rough warm start solution. If multiple `n_nodes_init` are specified,
        performs warm start with each of these. We require
        `3 <= n_nodes_init <= n_nodes`.
    tol : float, default=1e-06
        Convergence tolerance for the SLSQP optimizer.
    max_iter : int, default=500
        Maximum number of SLSQP iterations.
    t1_tol : float, default=1e-06
        Tolerance for the running cost when determining convergence of the
        finite horizon approximation.
    interp_tol : float, default=1e-03
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

    n_nodes = max(4, int(n_nodes))
    if n_nodes_init is None:
        n_nodes_init = n_nodes / 2
    n_nodes_init = np.clip(n_nodes_init, 3, n_nodes - 1)
    n_nodes_init = np.unique(n_nodes_init.astype(int))

    tol = max(float(tol), np.finfo(float).eps)
    t1_tol = max(float(t1_tol), np.finfo(float).eps)
    interp_tol = max(float(interp_tol), np.finfo(float).eps)

    max_n_segments = max(int(max_n_segments), 1)

    if isinstance(time_map, str):
        if time_map == 'log2':
            time_map = time_maps.TimeMapLog2
        elif time_map == 'rational':
            time_map = time_maps.TimeMapRational
        else:
            raise ValueError(f"time_map = {time_map} is not recognized. Valid "
                             f"options are 'log2' and 'rational'")

    f, converge_event, interp_event = _setup_open_loop(ocp, t1_tol, interp_tol)
    bound_events = _make_state_bound_events(ocp)

    events = (converge_event, interp_event)
    if bound_events is not None:
        events = events + tuple(bound_events)

    sols, t_break = [], []

    solve_kwargs = {'time_map': time_map, 'tol': tol, 'max_iter': max_iter,
                    'reshape_order': reshape_order, 'verbose': verbose}

    for k in range(max_n_segments):
        for n in n_nodes_init:
            warm_start_sol = _solve_infinite_horizon(ocp, t, x, u, n_nodes=n,
                                                     **solve_kwargs)

            if warm_start_sol.status in [0, 9]:
                # Accept successful solutions or solutions which maxed out the
                # allowed number of iterations
                t, x, u = warm_start_sol.t, warm_start_sol.x, warm_start_sol.u
            elif verbose:
                print("Ignoring failed warm start solution...")

        sols.append(_solve_infinite_horizon(ocp, t, x, u, n_nodes=n_nodes,
                                            **solve_kwargs))

        ode_sol = solve_ivp(f, [0., sols[-1].t[-1]], sols[-1].x[:, 0],
                            events=events, args=(sols[-1],),
                            exact_event_times=True, method=integration_method,
                            atol=atol, rtol=rtol, vectorized=True)

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
            if verbose:
                print("Terminating optimization: " + sols[-1].message)
            break

        # Otherwise, the interpolation error is greater than the tolerance, so
        # solve a new OCP
        # Sometimes when the ODE solution fails too early we run into problems,
        # so make sure we advance time at least a little
        t1 = np.maximum(ode_sol.t[-1], sols[-1].t[1])

        if len(t_break) >= 1:
            t_break.append(t1 + t_break[-1])
        else:
            t_break.append(t1)

        # Get the initial guess for the next segment
        t, x, u = _get_next_segment_guess(sols[-1], t1, ode_sol.y[:, -1:])

        if verbose:
            L1 = ocp.running_cost(x[:, 0], u[:, 0])
            print(f"Starting new Bellman segment at t{k + 1} = {t_break[-1]}")
            print(f"Running cost L(t{k + 1}) = {L1}")

    if len(sols) == 1:
        return sols[0]

    return CombinedSolution(sols, t_break)


def _solve_infinite_horizon(ocp, t, x, u, time_map=time_maps.TimeMapLog2,
                            n_nodes=32, tol=1e-06, max_iter=500,
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
    time_map : `TimeMapRadau`, default=`TimeMapLog2`
        `TimeMapRadau` subclass implementing `physical_to_radau`,
        `radau_to_physical`, and `derivative` methods. The default maps
        `radau_to_physical(tau) = log(4 / (1 - tau) ** 2)`.
    n_nodes : int, default=32
        Number of nodes to use in the pseudospectral discretization. Must be at
        least 3.
    tol : float, default=1e-06
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

    tau, w, D = radau.make_scaled_lgr(n_nodes,
                                      time_map_deriv=time_map.derivative)
    cost_fun, dyn_constr, bounds = setup_nlp.setup(ocp, x[:, 0], tau, w, D,
                                                   order=reshape_order)

    # Map initial guess to LGR points
    t_interp = time_map.radau_to_physical(tau)
    x, u = setup_nlp.interp_guess(t, x, u, t_interp)
    xu = setup_nlp.collect_vars(x, u, order=reshape_order)

    if verbose:
        print(f"\nNumber of LGR nodes: {n_nodes}")
        print("-" * 80)

    minimize_opts = {'maxiter': max_iter, 'iprint': verbose, 'disp': verbose}

    minimize_result = minimize(cost_fun, xu, bounds=bounds,
                               constraints=dyn_constr, jac=True, tol=tol,
                               options=minimize_opts)

    return DirectSolution.from_minimize_result(minimize_result, ocp, tau, w,
                                               time_map=time_map,
                                               order=reshape_order)


def _setup_open_loop(ocp, t1_tol, interp_tol):
    # Open-loop controlled system dynamics
    def dynamics(t, x, sol):
        u_interp = sol(t, return_x=False, return_p=False, return_v=False)
        if x.ndim < 2:
            u_interp = u_interp.reshape(-1,)
        return ocp.dynamics(x, u_interp)

    # Terminate integration for sufficiently small running cost
    def running_cost_converged(t, x, sol):
        u_interp = sol(t, return_x=False, return_p=False, return_v=False)
        if x.ndim < 2:
            u_interp = u_interp.reshape(-1,)
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


def _get_next_segment_guess(last_sol, t1, x1):
    idx = last_sol.t >= t1
    # Need at least two points
    idx[-2:] = True

    t = last_sol.t[idx]
    t1 = np.minimum(t1, t[0])

    # Set the initial condition to the one obtained by integration
    x = np.hstack((x1.reshape(-1, 1), last_sol.x[:, idx]))

    u1 = last_sol(t1, return_x=False, return_p=False, return_v=False)
    u = np.hstack((u1.reshape(-1, 1), last_sol.u[:, idx]))

    # Time starts at zero. Add a small constant to keep points distinct
    t = np.concatenate(([0.], t - t1 + 1e-07))

    return t, x, u