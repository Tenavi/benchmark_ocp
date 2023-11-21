import numpy as np
from tqdm import tqdm

from ._ivp import solve_ivp
from ..utilities import closed_loop_jacobian


def integrate_fixed_time(ocp, controller, x0, t_span, t_eval=None,
                         method='RK45', atol=1e-06, rtol=1e-03):
    """
    Integrate continuous-time system dynamics with a given feedback controller
    over a fixed time horizon for one initial condition.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0 : (`ocp.n_states`,) array
        Initial state.
    t_span : 2-tuple of floats
        Interval of integration `(t0, tf)`. The solver starts with `t[0]=t0` and
        integrates until it reaches `t[-1]=tf`.
    t_eval : array_like, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If `None` (default), use points selected by the solver.
    method : string or `OdeSolver`, default='RK45'
        See `scipy.integrate.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `scipy.integrate.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `scipy.integrate.solve_ivp`.

    Returns
    -------
    t : (n_points,) array
        Time points.
    x : (`ocp.n_states`, n_points) array
        System states at times `t`.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `t_span`.
            *  1: A termination event occurred.
    """
    def fun(t, x):
        u = controller(x)
        u = ocp._saturate(u)
        return ocp.dynamics(x, u)

    def jac(t, x):
        return closed_loop_jacobian(x, ocp.jac, controller)

    ode_sol = solve_ivp(fun, t_span, x0, jac=jac, events=ocp.integration_events,
                        t_eval=t_eval, vectorized=True, method=method,
                        rtol=rtol, atol=atol)

    return ode_sol.t, ode_sol.y, ode_sol.status


def integrate_to_converge(ocp, controller, x0, t_int, t_max, norm=2, ftol=1e-03,
                          method='RK45', atol=1e-06, rtol=1e-03):
    """
    Integrate continuous time system dynamics with a given feedback controller
    until a steady state is reached or a specified time horizon is exceeded.
    Integration starts at `t[0]=0` and continues over intervals of length
    `t_int` until a steady state is reached or `abs(t[-1]) >= abs(t_max)`.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0 : (`ocp.n_states`,) array
        Initial state.
    t_int : float
        Time interval to step integration over. This function internally calls
        `integrate_closed_loop` with `t_span=(t[-1], t[-1] + t_int)`. Can be
        negative.
    t_max : float
        Maximum time allowed for integration. Can be negative.
    norm : {1, 2, `np.inf`}, default=2
            Integration continues until `norm(f(x,u)) <= ftol`, where `f`
            denotes `ocp.dynamics` and `norm` specifies the norm used for
            this condition (l1, l2, or l-infinity).
    ftol : float or array_like, default=1e-03
        Tolerance for detecting system steady states. If `ftol` is array_like,
        then it must have shape `(ocp.n_states,)` and specifies a different
        convergence tolerance for each component of the dynamics. This overrides
        and ignores `norm` so that the convergence criteria becomes
        `all(f(x,u) <= ftol)`.
    method : string or `OdeSolver`, default='RK45'
        See `scipy.integrate.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `scipy.integrate.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `scipy.integrate.solve_ivp`.

    Returns
    -------
    t : (n_points,) array
        Time points.
    x : (`ocp.n_states`, n_points) array
        Values of the state at times `t`.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The system reached a steady state as determined by `ftol`.
            *  1: A termination event occurred.
            *  2: `t[-1]` exceeded `t_max`.
    """
    ftol = np.reshape(ftol, -1)

    if np.size(t_int) != 1:
        raise ValueError("t_int must be a float")

    if np.size(t_max) != 1:
        raise ValueError("t_max must be a float")

    if np.abs(t_int) > np.abs(t_max):
        raise ValueError("abs(t_int) must be less than or equal to abs(t_max)")

    if np.sign(t_int) != np.sign(t_max):
        raise ValueError("t_int and t_max must have the same sign")

    if norm not in (1, 2, np.inf):
        raise ValueError("norm must be one of {1, 2, np.inf}")

    if np.size(ftol) not in (1, ocp.n_states) or np.any(ftol <= 0.):
        raise ValueError("ftol must be a positive float or array_like")

    t = np.zeros(1)
    x = np.reshape(x0, (-1, 1))

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while True:
        # Simulate the closed-loop system
        t_new, x_new, status = integrate_fixed_time(
            ocp, controller, x[:, -1], (t[-1], t[-1] + t_int),
            method=method, atol=atol, rtol=rtol)

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
        elif np.linalg.norm(f, ord=norm) <= ftol:
            break

        # Time exceeds maximum time horizon
        if np.abs(t[-1]) >= np.abs(t_max):
            status = 2
            break

    return t, x, status


def monte_carlo_fixed_time(ocp, controller, x0, t_span, t_eval=None,
                           method='RK45', atol=1e-06, rtol=1e-03):
    """
    Wraps `integrate_fixed_time` to integrate continuous time system dynamics
    with a given feedback controller over a fixed time horizon for multiple
    initial conditions.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0 : (`ocp.n_states`, n_sims) array
        Initial states.
    t_span : 2-tuple of floats
        Interval of integration `(t0, tf)`. The solver starts with `t[0]=t0` and
        integrates until it reaches `t[-1]=tf`.
    t_eval : array_like, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If `None` (default), use points selected by the solver.
    method : string or `OdeSolver`, default='RK45'
        See `scipy.integrate.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `scipy.integrate.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `scipy.integrate.solve_ivp`.

    Returns
    -------
    sims : (n_sims,) object array of dicts
        The results of the closed loop simulations for each initial condition,
        `x0[:, i]`. Each list element is a dict containing

            * t : (n_points,) array
                Time points.
            * x : (`ocp.n_states`, n_points) array
                System states at times `t`.
            * u : (`ocp.n_controls`, n_points) array
                Feedback control inputs at times `t`.
    status : (n_sims,) integer array
        `status[i]` contains the reason for algorithm termination for `sims[i]`:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `t_span`.
            *  1: A termination event occurred.
    """
    return _monte_carlo(ocp, controller, x0, integrate_fixed_time, t_span,
                        t_eval=t_eval, method=method, atol=atol, rtol=rtol)


def monte_carlo_to_converge(ocp, controller, x0, t_int, t_max, norm=2,
                            ftol=1e-03, method='RK45', atol=1e-06, rtol=1e-03):
    """
    Wraps `integrate_to_converge` to integrate continuous-time system dynamics
    with a given feedback controller until a steady state is reached or a
    specified time horizon is exceeded.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0 : (`ocp.n_states`, n_sims) array
        Initial states.
    t_int : float
        Time interval to step integration over. This function internally calls
        `integrate_closed_loop` with `t_span=(t[-1], t[-1] + t_int)`.
    t_max : float
        Maximum time allowed for integration.
    norm : {1, 2, `np.inf`}, default=2
            Integration continues until `norm(f(x,u)) <= ftol`, where `f`
            denotes `ocp.dynamics` and `norm` specifies the norm used for
            this condition (l1, l2, or l-infinity).
    ftol : float or array_like, default=1e-03
        Tolerance for detecting system steady states. If `ftol` is array_like,
        then it must have shape `(ocp.n_states,)` and specifies a different
        convergence tolerance for each component of the dynamics. This overrides
        and ignores `norm` so that the convergence criteria becomes
        `all(f(x,u) <= ftol)`.
    method : string or `OdeSolver`, default='RK45'
        See `scipy.integrate.solve_ivp`.
    atol : float or array_like, default=1e-06
        See `scipy.integrate.solve_ivp`.
    rtol : float or array_like, default=1e-03
        See `scipy.integrate.solve_ivp`.

    Returns
    -------
    sims : (n_sims,) object array of dicts
        The results of the closed loop simulations for each initial condition,
        `x0[:, i]`. Each list element is a dict containing

            * 't' : (n_points,) array
                Time points.
            * 'x' : (`ocp.n_states`, n_points) array
                System states at times 't'.
            * 'u' : (`ocp.n_controls`, n_points) array
                Feedback control inputs at times 't'.
    status : (n_sims,) integer array
        `status[i]` contains the reason for algorithm termination for `sims[i]`:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `t_span`.
            *  1: A termination event occurred.
            *  2: `sims[i]['t'][-1]` exceeded `t_max`.
    """
    return _monte_carlo(ocp, controller, x0, integrate_to_converge, t_int,
                        t_max, norm=norm, ftol=ftol, method=method, atol=atol,
                        rtol=rtol)


def _monte_carlo(ocp, controller, x0_pool, fun, *args, **kwargs):
    x0_pool = np.reshape(x0_pool, (ocp.n_states, -1)).T
    n_sims = x0_pool.shape[0]

    sims = []
    status = np.zeros(n_sims, dtype=int)

    print(f"Simulating closed-loop system for {n_sims:d} initial conditions "
          f"({type(controller).__name__:s})...")
    for i in tqdm(range(n_sims)):
        t, x, status[i] = fun(ocp, controller, x0_pool[i], *args, **kwargs)
        sims.append({'t': t, 'x': x, 'u': controller(x)})

    return np.asarray(sims, dtype=object), status
