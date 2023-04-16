import numpy as np

from . import direct, indirect


def solve_fixed_time(ocp, t, x, u=None, p=None, v=None, method='indirect',
                     verbose=0, **kwargs):
    """
    Compute the open-loop optimal solution of a fixed time optimal control
    problem (OCP) for a single initial condition. This function allows using
    either a direct or indirect method (see below), or a user-supplied callable.

    If `method=='direct'` then this function calls `direct.solve_fixed_time`,
    which uses pseudospectral collocation to transform the OCP into a
    constrained optimization problem, which is then solved using sequential
    least squares quadratic programming (SLSQP).

    If `method=='indirect'` then this function wraps
    `indirect.solve_fixed_time`, which solves the two-point boundary value
    problem arising from Pontryagin's Maximum Principle.

    The direct method is generally considered to be more robust than the
    indirect method, which is known to be highly sensitive to the initial guess
    for the costates. On the other hand, when successful, the indirect method
    often yields more accurate results and can also be faster.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : `(n_points,)` array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : `(n_states, n_points)` array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    u : `(n_controls, n_points)` array, optional
        Initial guess for the optimal control at times `t`.  Required if
        `method=='direct'`.
    p : `(n_states, n_points)` array, optional
        Initial guess for the costate at times `t`. Required if
        `method=='indirect'`.
    v : `(n_points,)` array, optional
        Initial guess for the value function at states `x`.
    method : {'direct', 'indirect', callable}, default='indirect'
        Which solution method to use. If `method` is callable, then it will be
        called as
        `sol = method(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)`.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    **kwargs : dict
        Keyword arguments to pass to the solver. See `direct.solve_fixed_time`
        and `indirect.solve_fixed_time` for options.

    Returns
    -------
    sol : `OpenLoopSolution`
        Solution of the open-loop OCP. Should only be trusted if
        `sol.status == 0`.
    """
    if callable(method):
        return method(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)
    elif method == 'direct':
        if u is None:
            u = np.zeros((ocp.n_controls, np.size(t)))
        return direct.solve_fixed_time(ocp, t, x, u, verbose=verbose, **kwargs)
    elif method == 'indirect':
        if p is None:
            p = np.zeros_like(x)
        return indirect.solve_fixed_time(ocp, t, x, p, u=u, v=v,
                                         verbose=verbose, **kwargs)
    else:
        raise RuntimeError(f"method={method} is not one of the allowed options,"
                           f" 'direct', 'indirect', or callable")


def solve_infinite_horizon(ocp, t, x, u=None, p=None, v=None, method='indirect',
                           t1_tol=1e-10, verbose=0, **kwargs):
    """
    Compute the open-loop optimal solution of a finite horizon approximation of
    an infinite horizon optimal control problem (OCP) for a single initial
    condition. This function allows using either a direct or indirect method
    (see below), or a user-supplied callable.

    If `method=='direct'` then this function calls
    `direct.solve_infinite_horizon`, which uses pseudospectral collocation to
    transform the OCP into a constrained optimization problem, which is then
    solved using sequential least squares quadratic programming (SLSQP).

    If `method=='indirect'` then this function wraps
    `indirect.solve_infinite_horizon`, which solves the two-point boundary value
    problem arising from Pontryagin's Maximum Principle.

    The direct method is generally considered to be more robust than the
    indirect method, which is known to be highly sensitive to the initial guess
    for the costates. On the other hand, when successful, the indirect method
    often yields more accurate results and can also be faster. Both methods
    increase the length of the time horizon and the number of solution nodes
    until the running cost at final time is smaller than the desired tolerance.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : `(n_points,)` array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : `(n_states, n_points)` array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:, 0]`.
    u : `(n_controls, n_points)` array, optional
        Initial guess for the optimal control at times `t`. Required if
        `method=='direct'`.
    p : `(n_states, n_points)` array, optional
        Initial guess for the costate at times `t`. Required if
        `method=='indirect'`.
    v : `(n_points,)` array, optional
        Initial guess for the value function at states `x`.
    method : {'direct', 'indirect', callable}, default='indirect'
        Which solution method to use. If `method` is callable, then it will be
        called as
            ```
            sol = method(ocp, t, x, u=u, p=p, v=v, t1_tol=t1_tol,
                         verbose=verbose, **kwargs)
            ```
    t1_tol : float, default=1e-10
        Tolerance for the running cost when determining convergence of the
        finite horizon approximation.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    **kwargs : dict
        Keyword arguments to pass to the solver. See
        `direct.solve_infinite_horizon` and `indirect.solve_infinite_horizon`
        for options.

    Returns
    -------
    sol : `OpenLoopSolution`
        Solution of the open-loop OCP. Should only be trusted if
        `sol.status == 0`.
    """
    kwargs = {'t1_tol': t1_tol, 'verbose': verbose, **kwargs}
    if callable(method):
        return method(ocp, t, x, u=u, p=p, v=v, **kwargs)
    elif method == 'direct':
        if u is None:
            u = np.zeros((ocp.n_controls, np.size(t)))
        return direct.solve_infinite_horizon(ocp, t, x, u, **kwargs)
    elif method == 'indirect':
        if p is None:
            p = np.zeros_like(x)
        return indirect.solve_infinite_horizon(ocp, t, x, p, u=u, v=v, **kwargs)
    else:
        raise RuntimeError(f"method={method} is not one of the allowed options,"
                           f" 'direct', 'indirect', or callable")
