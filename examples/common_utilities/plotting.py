from itertools import combinations

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# Allow matplotlib to interpret LaTeX plot labels
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


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
                        x_labels=(), z_labels=(), fig_kwargs=dict()):
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
    x_labels : tuple, default=('$x_1$', '$x_2$', ...)
        Tuple of strings specifying how to label plot axes for states.
    z_labels : tuple, default=(f'${z}_1$', f'${z}_2$', ...)
        Tuple of strings specifying how to label the z-axis.
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

    x_labels = _check_labels(n_states, 'x', *x_labels)
    z_labels = _check_labels(n_other, z, *z_labels)

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

            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(x_labels[j])
            ax.set_zlabel(z_labels[k])
            ax.set_title(title)

    if len(figs) == 1:
        return figs[0]
    return figs


def plot_closed_loop(ocp, sims, x_labels=(), u_labels=(), subtitle=None,
                     fig_kwargs={}):
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
    x_labels : tuple, default=('$x_1$', '$x_2$', ...)
        Tuple of strings specifying how to label plot axes for states.
    u_labels : tuple, default=('$u_1$', '$u_2$', ...)
        Tuple of strings specifying how to label plot axes for controls.
    subtitle : str, optional
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

    x_labels = _check_labels(ocp.n_states, 'x', *x_labels)
    u_labels = _check_labels(ocp.n_controls, 'u', *u_labels)

    fig_kwargs = {'figsize': (6.4, n_plots * 1.5), **fig_kwargs}
    fig = plt.figure(**fig_kwargs)

    plt.subplots_adjust(hspace=0.5)

    for i in range(ocp.n_states):
        ax = plt.subplot(n_plots, 1, i + 1)
        ax.set_xlim(0., t_max)

        for sim in sims:
            ax.plot(sim['t'], sim['x'][i], 'k', alpha=0.5)

        ax.set_ylabel(x_labels[i])

        if i == 0:
            if subtitle is not None:
                ax.set_title(f'Closed-loop states ({subtitle})')
            else:
                ax.set_title('Closed-loop states')

    for i in range(ocp.n_controls):
        ax = plt.subplot(n_plots, 1, ocp.n_states + i + 1)
        ax.set_xlim(0., t_max)

        for sim in sims:
            ax.plot(sim['t'], sim['u'][i], 'k', alpha=0.5)

        ax.set_ylabel(u_labels[i])

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


def _check_labels(n_labels, backup_label, *labels):
    if len(labels) < n_labels:
        if n_labels == 1:
            labels = [f'${backup_label:s}$']
        else:
            new_labels = tuple(f'${backup_label:s}_{i + 1:d}$'
                               for i in range(n_labels))
            labels = labels + new_labels[len(labels):]

    return labels
