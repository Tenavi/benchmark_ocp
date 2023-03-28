import numpy as np

from . import direct, indirect


__all__ = ['solve_fixed_time', 'solve_infinite_horizon']


def solve_fixed_time(ocp, t, x, p=None, u=None, v=None, method='direct',
                     verbose=0, **kwargs):
    """
    Compute the open-loop optimal solution of a fixed time optimal control
    problem for a single initial condition. This function allows using either a
    direct or indirect method (see below), or a user-supplied callable.

    If `method=='direct'` then this function calls `direct.solve_fixed_time`,
    which uses pseudospectral collocation to transform optimal control problem
    into a constrained optimization problem, which is then solved using
    sequential least squares quadratic programming (SLSQP).

    If `method=='indirect'` then this function wraps
    `indirect.solve_fixed_time`, which solves the two-point boundary value
    problem (BVP) arising from Pontryagin's Maximum Principle (PMP). The
    underlying BVP solver is `scipy.integrate.solve_bvp`.

    The direct method is generally considered to be more robust than the
    indirect method, which is known to be highly sensitive to the initial guess
    for the costates. On the other hand, when successful, the indirect method
    often yields more accurate results and can also be faster.

    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    p : (n_states, n_points) array, optional
        Initial guess for the costate at times `t`. Required if
        `method=='indirect'`.
    u : (n_controls, n_points) array, optional
        Initial guess for the optimal control at times `t`.  Required if
        `method=='direct'`.
    v : (n_points,) array, optional
        Initial guess for the value function `v(x(t))`.
    method : {'direct', 'indirect', callable}, default='direct'
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
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:,0]`. Solution should only be
        trusted if `sol.status == 0`.
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


def solve_infinite_horizon(ocp, t, x, p=None, u=None, v=None, method='direct',
                           verbose=0, **kwargs):
    """
    Compute the open-loop optimal solution of a fixed time optimal control
    problem for a single initial condition. This function allows using either a
    direct or indirect method (see below), or a user-supplied callable.

    If `method=='direct'` then this function calls `direct.solve_fixed_time`,
    which uses pseudospectral collocation to transform optimal control problem
    into a constrained optimization problem, which is then solved using
    sequential least squares quadratic programming (SLSQP).

    If `method=='indirect'` then this function wraps
    `indirect.solve_fixed_time`, which solves the two-point boundary value
    problem (BVP) arising from Pontryagin's Maximum Principle (PMP). The
    underlying BVP solver is `scipy.integrate.solve_bvp`.

    The direct method is generally considered to be more robust than the
    indirect method, which is known to be highly sensitive to the initial guess
    for the costates. On the other hand, when successful, the indirect method
    often yields more accurate results and can also be faster.

    Parameters
    ----------
    ocp : OptimalControlProblem
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    t : (n_points,) array
        Time points at which the initial guess is supplied. Assumed to be
        sorted from smallest to largest.
    x : (n_states, n_points) array
        Initial guess for the state trajectory at times `t`. The initial
        condition is assumed to be contained in `x[:,0]`.
    p : (n_states, n_points) array, optional
        Initial guess for the costate at times `t`. Required if
        `method=='indirect'`.
    u : (n_controls, n_points) array, optional
        Initial guess for the optimal control at times `t`.  Required if
        `method=='direct'`.
    v : (n_points,) array, optional
        Initial guess for the value function `v(x(t))`.
    method : {'direct', 'indirect', callable}, default='direct'
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
    sol : OpenLoopSolution
        Bunch object containing the solution of the open-loop optimal control
        problem for the initial condition `x[:,0]`. Solution should only be
        trusted if `sol.status == 0`.
    """
    if callable(method):
        return method(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)
    elif method == 'direct':
        if u is None:
            u = np.zeros((ocp.n_controls, np.size(t)))
        return direct.solve_infinite_horizon(ocp, t, x, u, verbose=verbose,
                                             **kwargs)
    elif method == 'indirect':
        if p is None:
            p = np.zeros_like(x)
        return indirect.solve_infinite_horizon(ocp, t, x, p, u=u, v=v,
                                               verbose=verbose, **kwargs)
    else:
        raise RuntimeError(f"method={method} is not one of the allowed options,"
                           f" 'direct', 'indirect', or callable")
