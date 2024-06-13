import argparse as ap
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from examples.common_utilities.dynamics import quaternion_to_euler
from examples.common_utilities.plotting import make_legend, save_fig_dict
from examples.common_utilities import data_utils

from examples.uav.problem_definition import FixedWing
from examples.uav.fixed_wing_dynamics.containers import VehicleState


_control_labels = [r'$\delta_t$', r'$\delta_a$ [deg]',
                   r'$\delta_e$ [deg]', r'$\delta_r$ [deg]']
_control_scales = [1.] + [180. / np.pi] * 3
_pos_labels = [r'$p_n$ [m]', r'$p_e$ [m]', r'$h - h_f$ [m]']
_pos_labels_detail = [r'downrange [m]', r'crossrange [m]', r'altitude [m]']
_vel_labels = [r'$u$ [m/s]', r'$v$ [m/s]', r'$w$ [m/s]']
_eul_labels = [r'$\phi$ [deg]', r'$\theta$ [deg]',
               r'$\psi - \psi_f$ [deg]']
_pqr_labels = [r'$p$ [deg/s]', r'$q$ [deg/s]', r'$r$ [deg/s]']
_extra_labels = [r'$\chi - \chi_f$ [deg]', r'$\alpha$ [deg]',
                 r'$\beta$ [deg]', r'$\mathcal L$']


def plot_fixed_wing(ocp, sims, sim_labels=None, t_max=None,
                    x_min=None, x_max=None, fig_kwargs_ts={}, fig_kwargs_3d={},
                    save_dir=None):
    r"""
    Plot states, controls, and running cost vs. time for a set of trajectories,
    as well as a 3d plot of the UAV's path.

    Parameters
    ----------
    ocp : `FixedWing`
        Instance of the `FixedWing` problem associated with the simulation(s).
    sims : dict or list of dicts
        One or more time series, with each dict containing

            * 't' : (n_points,) array
                Time points.
            * 'x' : (n_states, n_points) array
                System states at times 't'.
            * 'u' : (n_controls, n_points) array
                Control inputs at times 't'.
    sim_labels : list of strings, optional
        Legend labels for each set of time series in `sims`.
    t_max : float, optional
        Optional upper limit for time axis for each time series plot.
    x_min : (15,) array, optional
        Optional lower limits for each state time series plot. The limits are
        assumed to be in the following order:
            $p_n, p_e, h - h_f, u, v, w, \phi, \theta, \psi - \psi_f, p, q, r$,
        and
            $\chi - \chi_f, \alpha, \beta$
        (course error, angle of attack, sideslip).
        Limits are assumed to be in [m] or [rad], depending on the state.
        Individual limits can be set to `None` to use the default based on data.
    x_max : (15,) array, optional
        Optional upper limits for each state time series plot.
    fig_kwargs_ts : dict, optional
        Keyword arguments to pass to the time series figure during creation. See
        `matplotlib.pyplot.figure`.
    fig_kwargs_3d : dict, optional
        Keyword arguments to pass to the 3d figure during creation. See
        `matplotlib.pyplot.figure`.
    save_dir : path_like, optional
        The directory where each figure should be saved. Figures will be saved
        as 'save_dir/time_series.pdf' and 'save_dir/3d.pdf'.

    Returns
    -------
    figs : list or None
        If `save_dir` is None, returns a list of two `Figure` instances with
        containing the simulation plots.
    """
    if isinstance(sims, dict):
        sims = [sims]

    states = [VehicleState.from_array(sim['x']) for sim in sims]
    positions = [_get_positions(sim['t'], state_traj)
                 for sim, state_traj in zip(sims, states)]

    if sim_labels is None or isinstance(sim_labels, str):
        sim_labels = [sim_labels] * len(sims)
    elif len(sim_labels) != len(sims):
        raise ValueError("If provided, len(sim_labels) must equal len(sims)")

    figs = {'time_series': _plot_time_series(ocp, sims, states, positions,
                                             sim_labels, t_max=t_max,
                                             x_min=x_min, x_max=x_max,
                                             fig_kwargs=fig_kwargs_ts),
            'flight_path': _plot_flight_path(positions, sim_labels,
                                             x_min=x_min, x_max=x_max,
                                             fig_kwargs=fig_kwargs_3d)}

    if save_dir is None:
        return figs.values()
    else:
        save_fig_dict(figs, save_dir, close_figs=True)


def _plot_time_series(ocp, sims, states, positions, sim_labels, t_max=None,
                      x_min=None, x_max=None, fig_kwargs={}):
    fig_kwargs = {'layout': 'constrained', 'figsize': (11, 4.8), **fig_kwargs}

    fig, axes = plt.subplots(nrows=4, ncols=5, **fig_kwargs)

    if t_max is None:
        t_max = np.max([sim['t'][-1] for sim in sims])

    if x_min is None:
        x_min = np.full([4, 4], None)
    else:
        x_min = np.reshape(x_min, (3, 5), order='F')
        x_min = np.vstack([x_min[:, :-1],
                           np.concatenate([x_min[:, -1], [None]])])
    if x_max is None:
        x_max = np.full([4, 4], None)
    else:
        x_max = np.reshape(x_max, (3, 5), order='F')
        x_max = np.vstack([x_max[:, :-1],
                           np.concatenate([x_max[:, -1], [None]])])

    for i in range(4):
        for j in range(5):
            ax = axes[i, j]
            ax.set_xlim(0., t_max)
            if i == 3:
                ax.set_xlabel('$t$ [s]', fontsize=12)

    # Plot controls
    for i in range(4):
        ax = axes[i, 0]

        for sim, label in zip(sims, sim_labels):
            ax.plot(sim['t'], sim['u'][i] * _control_scales[i], label=label)

        ax.set_ylim(ocp.control_lb[i] * _control_scales[i] - .01,
                    ocp.control_ub[i] * _control_scales[i] + .01)

        ax.set_ylabel(_control_labels[i], fontsize=12)

    # Plot states
    for sim, state_traj, pos, label in zip(sims, states, positions, sim_labels):
        t = sim['t']
        eul_angles = quaternion_to_euler(state_traj.attitude, degrees=True)
        # Change yaw, pitch, roll to roll, pitch, yaw order
        eul_angles = eul_angles[::-1]
        rates = np.rad2deg(state_traj.rates)

        for i in range(3):
            axes[i, 1].plot(t, pos[i], label=label)
            axes[i, 2].plot(t, state_traj.velocity[i], label=label)
            axes[i, 3].plot(t, eul_angles[i], label=label)
            axes[i, 4].plot(t, rates[i], label=label)

        axes[3, 1].plot(t, np.rad2deg(state_traj.course), label=label)
        axes[3, 2].plot(t, np.rad2deg(state_traj.airspeed[1]), label=label)
        axes[3, 3].plot(t, np.rad2deg(state_traj.airspeed[2]), label=label)
        axes[3, 4].plot(t, ocp.running_cost(sim['x'], sim['u']), label=label)

    axes[3, 4].set_yscale('log')

    for i in range(4):
        for j in range(4):
            axes[i, j + 1].set_ylim(x_min[i, j], x_max[i, j])

    for i in range(3):
        axes[i, 1].set_ylabel(_pos_labels[i], fontsize=12)
        axes[i, 2].set_ylabel(_vel_labels[i], fontsize=12)
        axes[i, 3].set_ylabel(_eul_labels[i], fontsize=12)
        axes[i, 4].set_ylabel(_pqr_labels[i], fontsize=12)

    for j in range(4):
        axes[3, 1 + j].set_ylabel(_extra_labels[j], fontsize=12)

    axes[0, 0].set_title('Controls', fontsize=14)
    axes[0, 1].set_title('Positions', fontsize=14)
    axes[0, 2].set_title('Velocities', fontsize=14)
    axes[0, 3].set_title('Attitude', fontsize=14)
    axes[0, 4].set_title('Rates', fontsize=14)
    axes[3, 1].set_title('Course', fontsize=14)
    axes[3, 2].set_title('Angle of attack', fontsize=14)
    axes[3, 3].set_title('Sideslip', fontsize=14)
    axes[3, 4].set_title('Running cost', fontsize=14)

    if any([label is not None for label in sim_labels]):
        make_legend(axes[0, 0], fontsize=12, loc='lower right')

    return fig


def _plot_flight_path(positions, sim_labels, x_min=None, x_max=None,
                      fig_kwargs={}):
    fig_kwargs = {'layout': 'constrained', **fig_kwargs}
    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Flight plath', fontsize=14)

    for pos, label in zip(positions, sim_labels):
        ax.plot(pos[1], pos[0], pos[2], label=label)

    if x_min is None or x_max is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        x_center = np.mean(xlim)
        y_center = np.mean(ylim)
        axis_size = np.max([xlim[1] - xlim[0],
                            ylim[1] - ylim[0],
                            zlim[1] - zlim[0]]) / 2.
        ax.set_xlim([x_center - axis_size, x_center + axis_size])
        ax.set_ylim([y_center - axis_size, y_center + axis_size])
        ax.set_zlim([-axis_size, axis_size])
    else:
        ax.set_xlim(x_min[1], x_max[1])
        ax.set_ylim(x_min[0], x_max[0])
        ax.set_zlim(x_min[2], x_max[2])

    ax.set_xlabel(_pos_labels_detail[1], fontsize=12)
    ax.set_ylabel(_pos_labels_detail[0], fontsize=12)
    ax.set_zlabel(_pos_labels_detail[2], fontsize=12)

    if any([label is not None for label in sim_labels]):
        make_legend(ax, fontsize=12)

    return fig


def _get_positions(t, states):
    d_pos = states.body_to_inertial(states.velocity)
    pos_fun = CubicSpline(t, d_pos, axis=1).antiderivative()
    pos = pos_fun(t)
    # Convert pd to altitude, and add initial altitude to normalize
    pos[-1] = -states.pd
    return pos


if __name__ == '__main__':
    from examples.uav import example_config as config

    parser = ap.ArgumentParser()

    parser.add_argument("-d", "--sim_data", type=str,
                        help="Path to a .csv file with closed-loop simulations "
                             "to plot and compare against open-loop solutions.")
    parser.add_argument("-c", "--ctrl_name", type=str,
                        help="Name of the controller used to generate the "
                             "closed-loop simulations in sim_data. If not "
                             "provided, the default uses the sim_data filename "
                             "without the extension, stripping '_sims' from "
                             "the end if included.")
    parser.add_argument("-s", "--show_plots", action='store_true',
                        help="If True, show plots.")
    args = parser.parse_args()

    ocp = FixedWing(**config.params)

    # Load and plot the open-loop optimal dataset
    data = data_utils.load_data(os.path.join(config.data_dir, 'data.csv'))

    if args.show_plots:
        plot_fixed_wing(ocp, data)
        plt.show()
    else:
        plot_fixed_wing(ocp, data,
                        save_dir=os.path.join(config.fig_dir, 'data'))

    # Load and plot closed-loop simulations
    if args.sim_data is not None:
        sim_data = data_utils.load_data(args.sim_data)

        # Get the default controller name from the data file name
        if args.ctrl_name is not None:
            ctrl_name = args.ctrl_name
        else:
            ctrl_name = Path(args.sim_data).stem.strip('_sims')

        # Loop through each closed-loop trajectory, assuming this corresponds to
        # the same open-loop optimal trajectory
        for i, sim in enumerate(sim_data):
            if args.show_plots:
                plot_fixed_wing(ocp, [sim_data[i], data[i]],
                                sim_labels=[ctrl_name, 'optimal'])
                plt.show()
            else:
                plot_fixed_wing(ocp, [sim_data[i], data[i]],
                                sim_labels=[ctrl_name, 'optimal'],
                                save_dir=os.path.join(config.fig_dir,
                                                      f'{ctrl_name}_sims',
                                                      f'sim_{i}'))
