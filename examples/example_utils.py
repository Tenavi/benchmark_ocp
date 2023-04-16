import numpy as np
import time
from tqdm import tqdm

import optimalcontrol as oc


def monte_carlo(ocp, controller, x0_pool, *args, **kwargs):
    sims = []
    success = []

    if np.isinf(ocp.final_time):
        int_fun = oc.simulate.integrate_to_converge
    else:
        int_fun = oc.simulate.integrate_fixed_time
        args = ([0., ocp.final_time],) + args

    for x0 in tqdm(x0_pool.T):
        t, x, status = int_fun(ocp, controller, x0, *args, **kwargs)
        sims.append({'t': t, 'x': x, 'u': controller(x)})
        success.append(status == 0)

    return sims, success


def generate_from_guess(ocp, guesses, verbose=0, **kwargs):
    """
    Given an existing open loop data set, resolve the open loop OCP using the
    previously generated data as initial guesses. Used when refining solutions
    with an indirect method or higher tolerances.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    guesses : list of `DataFrame`s
        list of initial guesses for each open-loop problem to be solved. Each
        element of `guesses` should be a `DataFrame` with columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' : Controls $u_1(t,x(t))$, ..., $u_m(t,x(t))$.
            * 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t)$.
            * 'v' : Value function/total cost $v(t,x(t))$.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    **kwargs : dict
        Keyword arguments to pass to the solver. For fixed final time problems,
        see `optimalcontrol.open_loop.solve_fixed_time`. For infinite horizon
        problem, see `optimalcontrol.open_loop.solve_infinite_horizon`.

    Returns
    -------
    data : list of `DataFrame`s
        Subset of `guesses` for which an acceptable solution was found. Each
        element is a `DataFrame` with the same structure as `guesses`.
    unsolved : list of `DataFrame`s
        Subset of `guesses` for which no acceptable solution was found. Each
        element is a `DataFrame` with the same structure as `guesses`,
        containing the solver's best attempt at a solution upon failure.
    success : list of bools
        List of same length as `guesses`. If `success[i] is True` then an
        acceptable solution based on `guesses[i]` was found and appended to
        `data`.
    sol_time : float
        Total time of successful solution attempts in seconds
    fail_time : float
        Total time of failed solution attempts in seconds
    """
    if np.isinf(ocp.final_time):
        sol_fun = oc.open_loop.solve_infinite_horizon
    else:
        sol_fun = oc.open_loop.solve_fixed_time

    if type(guesses) not in (list, np.ndarray):
        guesses = [guesses]

    data = []
    unsolved = []

    sol_time = []
    fail_time = []

    success = np.zeros(len(guesses), dtype=bool)

    print('\nSolving open loop optimal control problems...')
    w = str(len('attempted') + 2)
    row = '{:^' + w + '}|{:^' + w + '}|{:^' + w + '}'
    h1 = row.format('solved', 'attempted', 'desired')
    headers = ('\n' + h1, len(h1) * '-')

    for header in headers:
        print(header)

    for i, guess in enumerate(guesses):
        t, x, u, p, v = oc.utilities.unpack_dataframe(guess)

        start_time = time.time()

        sol = sol_fun(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)

        end_time = time.time()

        success[i] = sol.status == 0
        sol = oc.utilities.pack_dataframe(sol.t, sol.x, sol.u, sol.p, sol.v)

        if success[i]:
            sol_time.append(end_time - start_time)
            data.append(sol)
        else:
            fail_time.append(end_time - start_time)
            unsolved.append(sol)

        if verbose:
            for header in headers:
                print(header)
        print(row.format(len(data), i+1, len(guesses)),
              end='\r' if not verbose else None)

    sol_time, fail_time = np.sum(sol_time), np.sum(fail_time)

    return data, unsolved, success, sol_time, fail_time
