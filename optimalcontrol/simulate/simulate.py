import numpy as np

from ._ivp import solve_ivp
from ..utilities import closed_loop_jacobian


def integrate_closed_loop(ocp, controller, t_span, x0, t_eval=None,
                          method="RK45", atol=1e-06, rtol=1e-03):
    """
    Integrate continuous-time system dynamics with a given feedback controller
    over a fixed time horizon.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    controller : Controller
        An instance of a `Controller` subclass implementing `__call__` and
        `jacobian` methods.
    t_span : 2-tuple of floats
        Interval of integration `(t0, tf)`. The solver starts with `t=t0` and
        integrates until it reaches `t=tf`.
    x0 : (ocp.n_states,) array
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
    t : (n_points,) array
        Time points.
    x : (ocp.n_states, n_points) array
        Values of the state at times `t`.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `t_span`.
            *  1: A termination event occurred.
    """
    def fun(t, x):
        return ocp.dynamics(x, controller(x))

    def jac(t, x):
        return closed_loop_jacobian(x, ocp.jacobians, controller)

    ode_sol = solve_ivp(fun, t_span, x0, jac=jac, events=ocp.integration_events,
                        t_eval=t_eval, vectorized=True, method=method,
                        rtol=rtol, atol=atol)

    return ode_sol.t, ode_sol.y, ode_sol.status


def integrate_to_converge(ocp, controller, x0, t_int, t_max, norm=2, ftol=1e-03,
                          method="RK45", atol=1e-06, rtol=1e-03):
    """
    Integrate continuous-time system dynamics with a given feedback controller
    until a steady state is reached or a specified time horizon is exceeded.
    Integration starts at `t=0` and continues over intervals of length `t_int`
    until a steady state is reached or `t>=t_max`.

    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    controller : Controller
        An instance of a `Controller` subclass implementing `__call__` and
        `jacobian` methods.
    x0 : (ocp.n_states,) array
        Initial state.
    t_int : float
        Time interval to step integration over. This function internally calls
        `integrate_closed_loop` with `t_span=(t[-1], t[-1] + t_int)`.
    t_max : float
        Maximum time allowed for integration.
    norm : {1, 2, np.inf}, default=2
            Integration continues until `||f(x,u)|| <= ftol`, where `f()`
            denotes the system dynamics and `norm` specifies the norm used for
            this calculation (l1, l2, or infinity).
    ftol : float or array_like, default=1e-03
        Tolerance for detecting system steady states. If `ftol` is an
        `array_like`, then it must have shape `(ocp.n_states,)` and
        specifies a different convergence tolerance for each component of the
        state. This overrides and ignores `norm` so that the convergence
        criteria becomes `all(f(x,u) <= ftol)`.
    method : string or `OdeSolver`, default="RK45"
        See `simulate.ivp.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `simulate.ivp.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `simulate.ivp.solve_ivp`.

    Returns
    -------
    t : (n_points,) array
        Time points.
    x : (ocp.n_states, n_points) array
        Values of the state at times `t`.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The system reached a steady state as determined by `ftol`.
            *  1: A termination event occurred.
            *  2: `t` exceeded `t_max`.
    """
    ftol = np.reshape(ftol, -1)
    if np.size(ftol) not in (1,ocp.n_states) or np.any(ftol <= 0.):
        raise ValueError("ftol must be a positive float or array_like")

    if norm not in (1, 2, np.inf):
        raise ValueError("norm must be one of {1, 2, np.inf}")

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
            ocp, controller, (t[-1], t[-1] + t_int), x[:, -1],
            method=method, atol=atol, rtol=rtol
        )

        # Add new points to existing saved points. The first index of new points
        # duplicates the last index of existing points.
        t = np.concatenate((t, t_new[1:]))
        x = np.hstack((x, x_new[:, 1:]))

        # Integration fails
        if status != 0:
            break

        # System reaches steady state (status already is 0)
        f = ocp.dynamics(x[:, -1], controller(x[:, -1]))
        if ftol.shape[0] == ocp.n_states:
            if np.all(np.abs(f) <= ftol):
                break
        elif np.linalg.norm(f, ord=norm) < ftol:
            break

        # Time exceeds maximum time horizon
        if t[-1] >= t_max:
            status = 2
            break

    return t, x, status
