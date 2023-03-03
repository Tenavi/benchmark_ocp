import numpy as np

from .ivp import solve_ivp

def integrate_closed_loop(
        problem, controller, t_span, x0, t_eval=None,
        method="RK45", atol=1e-06, rtol=1e-03
    ):
    """
    Integrate continuous-time system dynamics with a given feedback controller
    over a fixed time horizon.

    Parameters
    ----------
    problem : object
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobian`, and `integration_events` methods.
    controller : object
        An instance of a `Controller` subclass implementing `__call__` and
        `jacobian` methods.
    t_span : 2-tuple of floats
        Interval of integration `(t0, tf)`. The solver starts with `t=t0` and
        integrates until it reaches `t=tf`.
    x0 : array_like, shape (problem.n_states,)
        Initial state.
    t_eval : array_like, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If `None` (default), use points selected by the solver.
    method : string or `OdeSolver`, default="RK45"
        See `simulate.ivp.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `simulate.ivp.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `simulate.ivp.solve_ivp`.

    Returns
    -------
    t : ndarray, shape (n_points,)
        Time points.
    x : ndarray, shape (problem.n_states, n_points)
        Values of the state at times `t`.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `t_span`.
            *  1: A termination event occurred.
    """
    def fun(t, x):
        return problem.dynamics(x, controller(x))

    def jac(t, x):
        return problem.jacobian(x, controller)

    ode_sol = solve_ivp(
        fun, t_span, x0, jac=jac, events=problem.integration_events,
        t_eval=t_eval, vectorized=True, method=method, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def integrate_to_converge(
        problem, controller, x0, t_int, t_max, ftol=1e-06,
        method="RK45", atol=1e-06, rtol=1e-03
    ):
    """
    Integrate continuous-time system dynamics with a given feedback controller
    until a steady state is reached or a specified time horizon is exceeded.
    Integration starts at `t=0` and continues over intervals of length `t_int`
    until a steady state is reached or `t>=t_max`.

    Parameters
    ----------
    problem : object
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobian`, and `integration_events` methods.
    controller : object
        An instance of a `Controller` subclass implementing `__call__` and
        `jacobian` methods.
    x0 : array_like, shape (problem.n_states,)
        Initial state.
    t_int : float
        Time interval to step integration over. This function internally calls
        `integrate_closed_loop` with `t_span=(t[-1], t[-1]+t_int)`.
    t_max : float
        Maximum time allowed for integration.
    ftol : float or array_like, default=1e-06
        Tolerance for detecting system steady states. Integration continues
        until all `abs(f(x(t),u(t))) <= ftol`, where `f()` denotes the system
        dynamics. If `ftol` is an array_like, then it must have shape
        `(problem.n_states,)` and specifies a different convergence tolerance
        for each component of the state.
    method : string or `OdeSolver`, default="RK45"
        See `simulate.ivp.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `simulate.ivp.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `simulate.ivp.solve_ivp`.

    Returns
    -------
    t : ndarray, shape (n_points,)
        Time points.
    x : ndarray, shape (problem.n_states, n_points)
        Values of the state at times `t`.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The system reached a steady state as determined by `ftol`.
            *  1: A termination event occurred.
            *  2: `t` exceeded `t_max`.
    """

    if np.size(ftol) not in (1,problem.n_states) or np.any(ftol <= 0.):
        raise ValueError("ftol must be a positive float or array_like")
    ftol = np.reshape(ftol, -1)

    if np.size(t_int) > 1 or t_int <= 0.:
        raise ValueError("t_int must be a positive float")

    if np.size(t_max) > 1 or t_max <= 0.:
        raise ValueError("t_max must be a positive float")

    if t_int > t_max:
        raise ValueError("t_int must be less than or equal to t_max")

    t = np.zeros(1)
    x = np.reshape(x0, (-1,1))

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while True:
        # Simulate the closed-loop system
        t_new, x_new, status = integrate_closed_loop(
            problem, controller, (t[-1], t[-1] + t_int), x[:,-1],
            method=method, atol=atol, rtol=rtol
        )

        # Add new points to existing saved points. The first index of new points
        # duplicates the last index of existing points.
        t = np.concatenate((t, t_new[1:]))
        x = np.hstack((x, x_new[:,1:]))

        # Integration fails
        if status != 0:
            break

        # System reaches steady state (status already is 0)
        f = problem.dynamics(x[:,-1], controller(x[:,-1]))
        if np.all(np.abs(f) <= ftol):
            break

        # Time exceeds maximum time horizon
        if t[-1] >= t_max:
            status = 2
            break

    return t, x, status
