import time

import numpy as np
from tqdm import tqdm

import optimalcontrol as oc


def monte_carlo(ocp, controller, x0_pool, fun, *args, **kwargs):
    """
    Wrapper of `optimalcontrol.simulate` for integrating closed-loop dynamics
    with multiple initial conditions.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jacobians`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and
        `jacobian` methods.
    x0_pool : (`ocp.n_states`, n_initial_conditions) array
        Set of initial states.
    fun : callable
        Function for integrating the closed-loop system. Usually
        `optimalcontrol.simulate.integrate_fixed_time` or
        `optimalcontrol.simulate.integrate_to_converge`, but may be any
        function with call signature
        ```
        t, x, status = fun(ocp, controller, x0, *args, **kwargs)
        ```
    *args : iterable
        Positional arguments to pass to `fun`.
    **kwargs : dict
        Keyword arguments to pass to `fun`.

    Returns
    -------
    sims : list of dicts
        The results of the closed loop simulations for each initial condition
        `x0_pool[:, i]`. Each list element is a dict containing

        * t : (n_points,) array
            Time points.
        * x : (`ocp.n_states`, n_points) array
            System states at times `t`.
        * u : (`ocp.n_controls`, n_data) array
            Control inputs at times `t`.
    success : list of bools
        Each element `success[i]` is `True` if the simulation for the initial
        conditions `x0_pool[:, i]` succeeded.
    """
    sims = []
    success = []

    x0_pool = np.reshape(x0_pool, (ocp.n_states, -1))

    for x0 in tqdm(x0_pool.T):
        t, x, status = fun(ocp, controller, x0, *args, **kwargs)
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
    guesses : list of DataFrames
        list of initial guesses for each open-loop problem to be solved. Each
        element of `guesses` should be a `DataFrame` with columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' : Controls $u_1(t,x(t))$, ..., $u_m(t,x(t))$.
            * 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t)$.
            * 'v' : Value function/cost-to-go $v(t,x(t))$.
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
    data : list of DataFrames
        Subset of `guesses` for which an acceptable solution was found. Each
        element is a `DataFrame` with the same structure as `guesses`.
    unsolved : list of DataFrames
        Subset of `guesses` for which no acceptable solution was found. Each
        element is a `DataFrame` with the same structure as `guesses`,
        containing the solver's best attempt at a solution upon failure.
    success : list of bools
        List of same length as `guesses`. If `success[i] is True` then an
        acceptable solution based on `guesses[i]` was found and appended to
        `data`.
    """
    if np.isinf(ocp.final_time):
        sol_fun = oc.open_loop.solve_infinite_horizon
    else:
        sol_fun = oc.open_loop.solve_fixed_time

    if type(guesses) not in (list, np.ndarray):
        guesses = [guesses]

    data = []
    unsolved = []

    sol_time = 0.
    fail_time = 0.

    success = np.zeros(len(guesses), dtype=bool)

    print("\nSolving open loop optimal control problems...")
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
            sol_time += end_time - start_time
            data.append(sol)
        else:
            fail_time += end_time - start_time
            unsolved.append(sol)

        if verbose:
            for header in headers:
                print(header)
        overwrite = not verbose and i+1 < len(guesses)
        print(row.format(len(data), i+1, len(guesses)),
              end='\r' if overwrite else None)

    print("\nTotal solution time:")
    print(f"    Successes: {sol_time:.1f} seconds")
    print(f"    Failures : {fail_time:.1f} seconds")

    return data, unsolved, success
