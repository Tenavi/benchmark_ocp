import numpy as np
import warnings
from copy import deepcopy

from .indirect import IndirectSolver
from .direct import DirectSolver


def solve_ocp(
        OCP, config,
        t_guess=None, X_guess=None, U_guess=None, dVdX_guess=None, V_guess=None,
        solve_to_converge=False, verbose=0, suppress_warnings=True
    ):
    if config.ocp_solver == 'indirect':
        solver = IndirectSolver(
            OCP,
            tol=config.indirect_tol,
            t1_scale=config.t1_scale,
            t1_max=config.t1_max,
            max_nodes=config.indirect_max_nodes
        )
    elif config.ocp_solver == 'direct':
        solver = DirectSolver(
            OCP,
            tol=config.direct_tol,
            tol_scale=config.direct_tol_scale,
            n_init_nodes=config.direct_n_init_nodes,
            n_add_nodes=config.direct_n_add_nodes,
            max_nodes=config.direct_max_nodes,
            max_iter=config.direct_max_slsqp_iter
        )
    elif config.ocp_solver == 'direct_to_indirect':
        config_copy = deepcopy(config)
        config_copy.ocp_solver = 'direct'

        direct_start, _, _ = solve_ocp(
            OCP,
            config_copy,
            t_guess=t_guess,
            X_guess=X_guess,
            U_guess=U_guess,
            dVdX_guess=dVdX_guess,
            V_guess=V_guess,
            solve_to_converge=True,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )

        config_copy.ocp_solver = 'indirect'

        idx = direct_start['t'] <= t_guess[-1]
        dVdX_guess = direct_start['dVdX'][:,idx]
        dVdX_guess[:,-1] = 0.

        return solve_ocp(
            OCP,
            config_copy,
            t_guess=direct_start['t'][idx],
            X_guess=direct_start['X'][:,idx],
            U_guess=direct_start['U'][:,idx],
            dVdX_guess=dVdX_guess,
            V_guess=direct_start['V'][:,idx],
            solve_to_converge=solve_to_converge,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )
    else:
        raise ValueError(
            'config.ocp_solver must be one of "direct", "indirect", or "direct_to_indirect"'
        )

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        solver.solve(
            t=t_guess, X=X_guess, U=U_guess, dVdX=dVdX_guess, V=V_guess,
            verbose=verbose
        )
        converged = solver.check_converged(config.fp_tol)

        # Solves the OCP over an extended time interval until convergece
        # conditions are satisfied
        if solve_to_converge and not converged:
            if suppress_warnings:
                # If we don't want to see warnings, treat these as errors so the
                # try context will catch them and mark the trajectory as not
                # converged instead of printing out the warning.
                warnings.filterwarnings("error", category=RuntimeWarning)

            try:
                while not converged and solver.extend_horizon():
                    solver.solve(**solver.sol, verbose=verbose)
                    converged = solver.check_converged(config.fp_tol)
            except RuntimeWarning:
                pass

        return solver.sol, solver.continuous_sol, converged


def clip_trajectory(t, X, U, criteria):
    '''
    Go backwards in time and check to see when a function (typically the
    running cost or vector field norm) is sufficiently small.

    Parameters
    ----------
    criteria : callable

    Returns
    -------
    converged_idx : int
        Integer such that X[:,converged_idx], U[:,converged_idx] is the first
        state-control pair for which criteria(X, U) < tol.
    converged : bool
        True if some pair X, U satisfied criteria(X, U) < tol, False if no
        such pair was found.
    '''
    converged_idx = criteria(X, U).flatten()

    converged = converged_idx.any()

    if converged:
        converged_idx = np.min(np.argwhere(converged_idx))
    else:
        converged_idx = t.shape[0] - 1

    return converged_idx, converged