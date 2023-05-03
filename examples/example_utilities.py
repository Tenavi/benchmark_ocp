import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from optimalcontrol import open_loop, controls, utilities


# Allow matplotlib to interpret LaTeX plot labels
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


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

        data.append(
            {'t': sol.t, 'x': sol.x, 'u': sol.u, 'p': sol.p, 'v': sol.v})

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


class NNController(controls.Controller):
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
        print("\nTraining neural network controller...")
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
            self.u_lb = utilities.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = utilities.resize_vector(self.u_ub, self.n_controls)

        self._train_time = time.time() - start_time
        print(f"\nTraining time: {self._train_time:.2f} seconds")

    def __call__(self, x):
        x_T = np.reshape(x, (self.n_states, -1)).T
        x_T = self._x_scaler.transform(x_T)

        u = self._regressor.predict(x_T).reshape(-1, self.n_controls)
        u = self._u_scaler.inverse_transform(u).T
        u = utilities.saturate(u, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u.flatten()

        return u


def make_legend(ax, **opts):
    """
    Make a legend without duplicate entries.

    Parameters
    ----------
    ax : pyplot.Axes
        `Axes` instance for which to add legend.
    **opts : dict, optional
        Keyword arguments to pass to the legend creation. Default arguments are
        `frameon=True`.

    Returns
    -------
    leg : matplotlib.Legend
        `Legend` instance created for `ax`.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    opts = {'frameon': True, **opts}
    leg = ax.legend(by_label.values(), by_label.keys(), **opts)
    return leg


def plot_total_cost(optimal_costs, controller_costs=dict(),
                    title='Closed-loop cost evaluation', fig_kwargs=dict()):
    """
    Plot the total accumulated cost for a set of closed-loop simulations with
    one or more feedback controls against the optimal value function. For an
    optimal feedback law, the plotted points should like exactly on the diagonal
    of the plot. In most cases, points should be above the diagonal, but they
    may fall below in cases where the open-loop solution is suboptimal or some
    other numerical problems have been encountered.

    Parameters
    ----------
    optimal_costs : (n_points,) array_like
        The open-loop optimal value function at a set of initial conditions.
    controller_costs : dict
        Each key should be a string which is the name of a feedback control law
        and each value is an `(n_points,)` array_like containing the total cost
        as evaluated in closed-loop simulation for the same initial conditions
        as in `optimal_costs`.
    title : str, default='Closed-loop cost evaluation'
        Title for the figure.
    fig_kwargs : dict, optional
        Keyword arguments to pass during figure creation. See
        `matplotlib.pyplot.figure`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure instance with a scatterplot of closed-loop vs. optimal costs.
    """
    fig = plt.figure(**fig_kwargs)

    ax = plt.axes()

    for label, costs in controller_costs.items():
        plt.scatter(optimal_costs, costs, s=16, label=label)

    ax.set_ylim(bottom=ax.get_xlim()[0])
    ax.set_xlim(right=ax.get_ylim()[1])

    plt.plot(ax.get_xlim(), ax.get_ylim(), 'k--')

    ax.set_xlabel('optimal value $V$')
    ax.set_ylabel('closed-loop cost $J$')
    ax.set_title(title)

    make_legend(ax)

    return fig


def plot_closed_loop_3d(sims, open_loop_sols, z='u',
                        controller_name='learning-based control',
                        title='Closed-loop trajectories and controls',
                        fig_kwargs=dict()):
    """
    Plot closed-loop simulations and open-loop solutions together on a single 3d
    plot. This produces one figure for each pairwise combination of system
    states and each specified z-axis variable (e.g. controls).

    Parameters
    ----------
    sims : length n_sims list of dicts
        Closed loop simulations output by
        `optimalcontrol.simulate.monte_carlo_fixed_time` or
        `optimalcontrol.simulate.monte_carlo_to_converge`. Each element of
        `sims` should be a dict with keys

            * 't' : (n_points,) array
                Time points.
            * 'x' : (n_states, n_points) array
                System states at times 't'.
            * `z` : (n_other, n_points) array
                Third variable (e.g. control inputs 'u') at times 't'.
    open_loop_sols : length n_sims list of dicts
        Solutions of the open loop OCP for each initial condition in `sims`.
        Each element of `open_loops_sols` should be a dict with the same keys
        as `sims`.
    z : str, default='u'
        Which variable (default='u', controls) to plot on the z-axis. Must be
        a key in each dict contained in `sims`.
    controller_name : str, default='learning-based controller'
        How the feedback controller should be referred to in plots.
    title : str, default='Closed-loop trajectories and controls'
        The title for each figure.
    fig_kwargs : dict, optional
        Keyword arguments to pass during figure creation. See
        `matplotlib.pyplot.figure`.

    Returns
    -------
    figs : (list of) `matplotlib.figure.Figure`(s)
        Each list element contains a figure instance which plots a combination
        of states and `z`.
    """
    figs = list()

    x_all = np.hstack([sim['x'] for sim in sims])
    z_all = np.hstack([sim[z] for sim in sims])

    n_states = x_all.shape[0]
    n_other = z_all.shape[0]

    if n_states < 2:
        raise ValueError("plot_closed_loop_3d is only implemented for "
                         "n_states >= 2")

    for i, j in combinations(range(n_states), 2):
        for k in range(n_other):
            figs.append(plt.figure(**fig_kwargs))

            ax = plt.axes(projection='3d')

            for sol in open_loop_sols:
                ax.plot(sol['x'][i], sol['x'][j], sol['u'][k], 'k', alpha=0.5,
                        label='open-loop optimal')

            ax.scatter(x_all[i], x_all[j], z_all[k], c=z_all[k], marker='o',
                       s=9, alpha=0.5, label=controller_name)

            ax.set_xlabel(f'$x_{i+1:d}$')
            ax.set_ylabel(f'$x_{j+1:d}$')
            ax.set_zlabel(f'${z:s}_{k+1:d}$' if n_other > 1 else f'${z:s}$')
            ax.set_title(title)

    if len(figs) == 1:
        return figs[0]
    return figs


def plot_closed_loop(ocp, sims, data_name=None, fig_kwargs=dict()):
    """
    Plot the states, controls, and running cost vs. time for a set of
    trajectories.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing a
        `running_cost` method.
    sims : length n_sims list of dicts
        Closed loop simulations output by
        `optimalcontrol.simulate.monte_carlo_fixed_time` or
        `optimalcontrol.simulate.monte_carlo_to_converge`. Each element of
        `sims` should be a dict with keys

            * 't' : (n_points,) array
                Time points.
            * 'x' : (`ocp.n_states`, n_points) array
                System states at times 't'.
            * 'u' : (`ocp.n_controls`, n_points) array
                Control inputs at times 't'.
    data_name : str, optional
        If provided, this string appears in parentheses after the first plot
        title.
    fig_kwargs : dict, optional
        fig_kwargs : dict, optional
        Keyword arguments to pass during figure creation. See
        `matplotlib.pyplot.figure`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure instance with a set of plots of each state, control, and the
        running cost vs. time for all trajectories.
    """
    t_max = np.max([sim['t'][-1] for sim in sims])

    n_plots = ocp.n_states + ocp.n_controls + 1

    fig_kwargs = {'figsize': (6.4, n_plots * 1.5), **fig_kwargs}
    fig = plt.figure(**fig_kwargs)

    plt.subplots_adjust(hspace=0.5)

    for i in range(ocp.n_states):
        ax = plt.subplot(n_plots, 1, i + 1)
        ax.set_xlim(0., t_max)

        for sim in sims:
            ax.plot(sim['t'], sim['x'][i], 'k', alpha=0.5)

        ax.set_ylabel(f'$x_{i+1:d}$' if ocp.n_states > 1 else f'$x$')

        if i == 0:
            if data_name is not None:
                ax.set_title(f'Closed-loop states ({data_name})')
            else:
                ax.set_title('Closed-loop states')

    for i in range(ocp.n_controls):
        ax = plt.subplot(n_plots, 1, ocp.n_states + i + 1)
        ax.set_xlim(0., t_max)

        for sim in sims:
            ax.plot(sim['t'], sim['u'][i], 'k', alpha=0.5)

        ax.set_ylabel(f'$u_{i + 1:d}$' if ocp.n_controls > 1 else f'$u$')

        if i == 0:
            ax.set_title('Feedback controls')

    ax = plt.subplot(n_plots, 1, n_plots)
    ax.set_xlim(0., t_max)

    for sim in sims:
        ax.plot(sim['t'], ocp.running_cost(sim['x'], sim['u']), 'k', alpha=0.5)

    ax.set_yscale('log')

    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\mathcal L$')
    ax.set_title('Running cost')

    return fig
