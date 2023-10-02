import time

import numpy as np
import pandas as pd

from optimalcontrol import open_loop, utilities


def generate_data(ocp, guesses, verbose=0, **kwargs):
    """
    Given an existing open loop data set, resolve the open loop OCP using the
    previously generated data as initial guesses. Used when refining solutions
    with an indirect method or higher tolerances.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `optimal_control` methods.
    guesses : length n_problems list of dicts,
        Initial guesses for each open-loop OCP. Each element of `guesses` should
        be a dict or DataFrame with keys

            * t : (n_points,) array
                Time points.
            * x : (`ocp.n_states`, n_points) array
                Guess for system states at times `t`.
            * u : (`ocp.n_controls`, n_points) array, optional
                Guess for optimal controls at times `t`. Required if
                `method=='direct'`.
            * p : (`ocp.n_states`, n_points) array, optional
                Guess for costates at times `t`. Required if
                `method=='indirect'` (default).
            * v : (n_points,) array, optional
                Guess for value function at times `t`.
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
    data : (n_problems,) object array of dicts
        Solutions or attempted solutions of the open loop OCP based on
        `guesses`. Each element is a dict with the same keys and values as
        `guesses`. If `status[i]==0`, then `data[i]` is considered an acceptable
        solution, otherwise `data[i]` contains the solver's best attempt at a
        solution upon failure (which may simply be the original `guesses[i]`).
    status : (n_problems,) integer array
        `status[i]` contains an int describing if an acceptable solution based
        on `guesses[i]` was found. In particular, if `status[i]==0` then
        `data[i]` was deemed acceptable.
    messages : (n_problems,) string array
        `messages[i]` contains a human-readable message describing `status[i]`.
    """
    if np.isinf(ocp.final_time):
        sol_fun = open_loop.solve_infinite_horizon
    else:
        sol_fun = open_loop.solve_fixed_time

    if type(guesses) not in (list, np.ndarray):
        guesses = [guesses]

    data = []
    status = np.zeros(len(guesses), dtype=int)
    messages = []

    n_succeed = 0

    sol_time = 0.
    fail_time = 0.

    print("\nSolving open loop optimal control problems...")
    w = str(len('attempted') + 2)
    row = '{:^' + w + '}|{:^' + w + '}|{:^' + w + '}'
    h1 = row.format('solved', 'attempted', 'desired')
    headers = ('\n' + h1, len(h1) * '-')

    for header in headers:
        print(header)

    for i, guess in enumerate(guesses):
        t, x, u, p, v = utilities.unpack_dataframe(guess)

        start_time = time.time()

        sol = sol_fun(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)

        end_time = time.time()

        status[i] = sol.status
        messages.append(sol.message)

        if status[i] == 0:
            sol_time += end_time - start_time
            n_succeed += 1
        else:
            fail_time += end_time - start_time

        data.append({'t': sol.t,
                     'x': sol.x,
                     'u': sol.u,
                     'p': sol.p,
                     'v': sol.v,
                     'L': ocp.running_cost(sol.x, sol.u)})

        if verbose:
            for header in headers:
                print(header)
        overwrite = not verbose and i + 1 < len(guesses)
        print(row.format(n_succeed, i + 1, len(guesses)),
              end='\r' if overwrite else None)

    print("\nTotal solution time:")
    print(f"    Successes: {sol_time:.1f} seconds")
    print(f"    Failures : {fail_time:.1f} seconds")

    if n_succeed < len(guesses):
        print("\nFailed initial conditions:")
        for i, stat in enumerate(status):
            if stat != 0:
                print(f"i={i:d} : status = {stat:d} : {messages[i]:s}")

    data = np.asarray(data, dtype=object)
    messages = np.asarray(messages, dtype=str)

    return data, status, messages


def save_data(data, filepath):
    """
    Save a dataset of open-loop optimal control solutions or closed-loop
    simulations to a csv file. The data will be saved as a single csv columns
    with all trajectories concatenated vertically. A dataset saved in this
    format can be recovered by `load_data`.

    Parameters
    ----------
    data : list of dicts or DataFrames
        Each element of `data_list` is a dict or `DataFrame` with keys/columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' or 'x' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' or 'u' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p1', ..., 'pn' or 'p' : Costates $p_1(t)$, ..., $p_n(t)$,
                optional.
            * 'v' : Value function/cost-to-go $v(t)$, optional.
    filepath : path-like
        Where the csv file should be saved.
    """
    t, x, u, p, v = utilities.stack_dataframes(*data)
    data = utilities.pack_dataframe(t, x, u, p, v)
    data.to_csv(filepath, index=False)


def load_data(filepath):
    """
    Load a dataset of open-loop optimal control solutions or closed-loop
    simulations from a csv file. To break apart the dataset, assumes that the
    csv file contains vertically concatenated trajectories with initial time
    `t==0`.

    Parameters
    ----------
    filepath : path-like
        Where the csv file should be saved.

    Returns
    -------
    data : list of DataFrames
        Each element of `data` is a `DataFrame` with columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t).
            * 'v' : Value function/cost-to-go $v(t)$.
    """
    dataframe = pd.read_csv(filepath)
    # Find where trajectories start
    t0_idx = np.where(dataframe['t'].to_numpy() == 0.)[0]
    # Assume trajectories end before the start of the next trajectory
    # Pandas includes the ends of index slices, so subract 1 from these
    t1_idx = np.concatenate((t0_idx[1:] - 1, [len(dataframe)]))
    data = []
    for i0, i1 in zip(t0_idx, t1_idx):
        data.append(dataframe.loc[i0:i1].reset_index(drop=True))
    return data
