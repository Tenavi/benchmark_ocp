import time

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

import optimalcontrol as oc


def monte_carlo(ocp, controller, x0_pool, fun, *args, **kwargs):
    """
    Wrapper of `optimalcontrol.simulate` for integrating closed-loop dynamics
    with multiple initial conditions.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
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
        * u : (`ocp.n_controls`, n_points) array
            Control inputs at times `t`.
    success : list of bools
        Each element `success[i]` is `True` if the simulation for the initial
        conditions `x0_pool[:, i]` succeeded.
    """
    sims = []
    success = []

    x0_pool = np.reshape(x0_pool, (ocp.n_states, -1)).T

    print("\nIntegrating closed-loop dynamics...")
    for x0 in tqdm(x0_pool):
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
    guesses : list of DataFrames, length n_problems
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
    data : list of DataFrames, length n_problems
        Solutions or attempted solutions of the open loop OCP based on
        `guesses`. Each element is a `DataFrame` with the same structure as
        `guesses`. If `status[i]==0`, then `data[i]` is considered an acceptable
        solution, otherwise `data[i]` contains the solver's best attempt at a
        solution upon failure (which may simply be the original `guesses[i]`).
    status : (n_problems,) integer array
        `status[i]` contains an int describing if an acceptable solution based
        on `guesses[i]` was found. In particular, if `status[i]==0` then
        `data[i]` was deemed acceptable.
    messages : list of strings, length n_problems
        `messages[i]` contains a human-readable message describing `status[i]`.
    """
    if np.isinf(ocp.final_time):
        sol_fun = oc.open_loop.solve_infinite_horizon
    else:
        sol_fun = oc.open_loop.solve_fixed_time

    if type(guesses) not in (list, np.ndarray):
        guesses = [guesses]

    data = []
    status = - np.ones(len(guesses), dtype=int)
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
        t, x, u, p, v = oc.utilities.unpack_dataframe(guess)

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

        data.append(
            oc.utilities.pack_dataframe(sol.t, sol.x, sol.u, sol.p, sol.v))

        if verbose:
            for header in headers:
                print(header)
        overwrite = not verbose and i+1 < len(guesses)
        print(row.format(n_succeed, i+1, len(guesses)),
              end='\r' if overwrite else None)

    print("\nTotal solution time:")
    print(f"    Successes: {sol_time:.1f} seconds")
    print(f"    Failures : {fail_time:.1f} seconds")

    if n_succeed < len(guesses):
        print("\nFailed initial conditions:")
        for i, stat in enumerate(status):
            if stat != 0:
                print(f"i={i:d} : status = {stat:d} : {messages[i]:s}")

    return data, status, messages


def evaluate_closed_loop(ocp, controller, x0_pool, data, lqr_sims, data_idx,
                         data_name, fun, *args, **kwargs):
    """
    Evaluate a learned controller in closed-loop simulation. Currently only
    implemented for two states and one control input.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics`, `jac`, and `integration_events` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0_pool : (`ocp.n_states`, n_initial_conditions) array
        Set of initial states.
    data : list of DataFrames, length n_initial_conditions
        Solutions of the open loop OCP for each initial condition in `x0_pool`.
        Each element of `data` should be a `DataFrame` with columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' : Controls $u_1(t,x(t))$, ..., $u_m(t,x(t))$.
            * 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t)$.
            * 'v' : Value function/cost-to-go $v(t,x(t))$.
    lqr_sims : list of dicts, length n_initial_conditions
        LQR-in-the-loop simulations for each initial condition in `x0_pool`,
        output in the format of `monte_carlo`. Each element of `lqr_sim` should
        be a dict with keys

            * t : (n_points,) array
                Time points.
            * x : (`ocp.n_states`, n_points) array
                System states at times `t`.
            * u : (`ocp.n_controls`, n_points) array
                Control inputs at times `t`.
    data_idx : array of ints
        Indices of the subset of `x0_pool`, `data`, and `lqr_sims` to analyze.
    data_name : str
        How to refer to the subset of the data indicated by `data_idx`, e.g.
        'training'.
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
        `x0_pool[:, data_idx[i]]`. Each list element is a dict containing

        * t : (n_points,) array
            Time points.
        * x : (`ocp.n_states`, n_points) array
            System states at times `t`.
        * u : (`ocp.n_controls`, n_points) array
            Control inputs at times `t`.
    success : list of bools
        Each element `success[i]` is `True` if the simulation for the initial
        conditions `x0_pool[:, data_idx[i]]` succeeded.
    """
    if ocp.n_states != 2 or ocp.n_controls != 1:
        raise NotImplementedError

    sims, converged = monte_carlo(ocp, controller, x0_pool[:, data_idx], fun,
                                  *args, **kwargs)

    for sim in sims:
        sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]

    n_converged = np.sum(converged)
    print(f"{n_converged}/{len(sims)} {data_name} trajectories practically "
          f"stabilized")

    optimal_cost = [sol.loc[0, 'v'] for sol in data[data_idx]]
    lqr_cost = [sim['v'][0] for sim in lqr_sims[data_idx]]
    controller_cost = [sim['v'][0] for sim in sims]

    plt.figure()

    ax = plt.axes()

    plt.scatter(optimal_cost, controller_cost, s=9,
                label='learning-based controller')
    plt.scatter(optimal_cost, lqr_cost, marker='x', label='LQR')

    ax.set_ylim(bottom=ax.get_xlim()[0])
    ax.set_xlim(right=ax.get_ylim()[1])

    plt.plot(ax.get_xlim(), ax.get_ylim(), 'k--', label='optimal feedback')

    ax.set_xlabel('Optimal value $V$')
    ax.set_ylabel('Closed-loop cost $J$')
    ax.set_title(f'Closed-loop cost evaluation ({data_name})')

    plt.legend()

    plt.figure()

    ax = plt.axes(projection='3d')

    for i, j in enumerate(data_idx):
        ax.plot(data[j]['x1'], data[j]['x2'], data[j]['u1'], 'k', alpha=0.5,
                label='open-loop optimal')

    x = np.hstack([sim['x'] for sim in sims])
    u = np.hstack([sim['u'] for sim in sims])
    ax.scatter(x[0], x[1], u[0], c=u, marker='o', s=9, alpha=0.5,
               label='learning-based controller')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$u$')
    ax.set_title(f'Closed-loop trajectories and controls ({data_name})')

    # Make legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True)

    plt.figure()

    ax = plt.axes()

    xf = np.hstack([sim['x'][:, -1:] for sim in sims])
    ax.scatter(xf[0], xf[1], s=9, alpha=0.5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Final positions ({data_name})')

    return sims


class NNController(oc.controls.Controller):
    """
    A simple example of how one might implement a neural network controller
    trained by supervised learning. To do this, we generate a dataset of
    state-control pairs (`x_data`, `u_data`), and optionally for time-dependent
    problems associated time values `t_data`, and the neural network learns the
    mapping from `x_data` (and `t_data`) to `u_data`. The neural network for
    this controller is implemented with `sklearn.neural_network.MLPRegressor`.
    """
    def __init__(self, x_data, u_data, t_data=None, u_lb=None, u_ub=None,
                 **options):
        """
        Parameters
        ----------
        x_data : (n_states, n_data) array
            A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        u_data : (n_contrls, n_data) array
            The optimal feedback controls evaluated at the states `x_data`.
        A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        t_data : (n_data,) array, optional
            For time-dependent problems, the time values at which the pairs
            (`x_data`, `u_data`) are obtained.
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            Upper control saturation bounds.
        **options : dict
            Keyword arguments to pass to `sklearn.neural_network.MLPRegressor`.
        """
        print("Training neural network controller...")
        start_time = time.time()

        x_data = np.transpose(np.atleast_2d(x_data))
        if t_data is not None:
            raise NotImplementedError("Time-dependent problems are not yet "
                                      "implemented")
        u_data = np.transpose(np.atleast_2d(u_data))

        # Scale the input and output data based on the interquartile range
        self._x_scaler = RobustScaler().fit(x_data)
        self._u_scaler = RobustScaler().fit(u_data)

        self.n_states = self._x_scaler.n_features_in_
        self.n_controls = self._u_scaler.n_features_in_

        # Fit the neural network to the data
        self._regressor = MLPRegressor(**options)
        self._regressor.fit(self._x_scaler.transform(x_data),
                            np.squeeze(self._u_scaler.transform(u_data)))

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = oc.utilities.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = oc.utilities.resize_vector(self.u_ub, self.n_controls)

        self._train_time = time.time() - start_time
        print(f"\nTraining time: {self._train_time:.2f} seconds")

    def __call__(self, x):
        x_T = np.reshape(x, (self.n_states, -1)).T
        x_T = self._x_scaler.transform(x_T)

        u = self._regressor.predict(x_T).reshape(-1, self.n_controls)
        u = self._u_scaler.inverse_transform(u).T
        u = oc.utilities.saturate(u, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u.flatten()

        return u
