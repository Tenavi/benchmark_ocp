import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from examples.common_utilities.dynamics import quaternion_to_euler
from examples.common_utilities.plotting import make_legend

from examples.uav.fixed_wing_dynamics.containers import VehicleState


_control_labels = [r'$\delta_t$', r'$\delta_a$ [$\circ$]',
                   r'$\delta_e$ [$\circ$]', r'$\delta_r$ [$\circ$]']
_control_scales = [1.] + [180. / np.pi] * 3
_pos_labels = [r'$p_n$ [m]', r'$p_e$ [m]', r'$h - h_f$ [m]']
_pos_labels_detail = [r'downrange $p_n$ [m]', r'crossrange $p_e$ [m]',
                      r'altitude $h - h_f$ [m]']
_vel_labels = [r'$u$ [m/s]', r'$v$ [m/s]', r'$w$ [m/s]']
_eul_labels = [r'$\phi$ [$\circ$]', r'$\theta$ [$\circ$]',
               r'$\psi - \psi_f$ [$\circ$]']
_pqr_labels = [r'$p$ [$\circ$/s]', r'$q$ [$\circ$/s]', r'$r$ [$\circ$/s]']
_extra_labels = [r'$\chi - \chi_f$ [$\circ$]', r'$\alpha$ [$\circ$]',
                 r'$\beta$ [$\circ$]', r'$\mathcal L$']
_title_props = {'fontsize': 14, 'fontweight': 'extra bold'}


def plot_closed_loop(sims, ocp, sim_labels=None, x_min=None, x_max=None,
                     fig_kwargs_ts={}, fig_kwargs_3d={}, save_dir=None):
    r"""
    Plot states, controls, and running cost vs. time for a set of trajectories.
    For the `BurgersPDE` problem, states are visualized as a heatmap.

    Parameters
    ----------
    sims : dict or list of dicts
        One or more time series, with each dict containing

            * 't' : (n_points,) array
                Time points.
            * 'x' : (n_states, n_points) array
                System states at times 't'.
            * 'u' : (n_controls, n_points) array
                Control inputs at times 't'.
            * 'L' : (n_points,) array, optional
                Running cost at times 't'.
    ocp : `FixedWing`
        Instance of the `FixedWing` problem associated with the simulation(s).
    sim_labels : list of strings, optional
        Legend labels for each set of time series in `sims`.
    x_min : (15,) array, optional
        Lower limits for each state plot. The limits are assumed to be in the
        following order:
            $p_n, p_e, p_d, u, v, w, \phi, \theta, \psi, p, q, r, \chi, \alpha, \beta$
        Limits are assumed to be in [m] or [rad], depending on the state.
    x_max : (15,) array, optional
        Upper limits for each state plot.
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

    t_max = np.max([sim['t'][-1] for sim in sims])

    states = [VehicleState.from_array(sim['x']) for sim in sims]

    if sim_labels is None or isinstance(sim_labels, str):
        sim_labels = [sim_labels] * len(sims)
    elif len(sim_labels) != len(sims):
        raise ValueError("If provided, len(sim_labels) must equal len(sims)")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    fig_kwargs_ts = {'layout': 'constrained', 'figsize': (11, 4.8),
                     **fig_kwargs_ts}
    fig_kwargs_3d = {'layout': 'constrained', **fig_kwargs_3d}

    fig_ts, axes = plt.subplots(nrows=4, ncols=5, **fig_kwargs_ts)

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
    for state_traj, sim, label in zip(states, sims, sim_labels):
        t = sim['t']
        pos = get_positions(sim['t'], state_traj)
        pos[-1] = -state_traj.pd
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
        axes[3, 4].plot(t, sim['L'], label=label)

    axes[3, 4].set_yscale('log')

    for i in range(3):
        axes[i, 1].set_ylabel(_pos_labels[i], fontsize=12)
        axes[i, 2].set_ylabel(_vel_labels[i], fontsize=12)
        axes[i, 3].set_ylabel(_eul_labels[i], fontsize=12)
        axes[i, 4].set_ylabel(_pqr_labels[i], fontsize=12)

    for j in range(4):
        axes[3, 1 + j].set_ylabel(_extra_labels[j], fontsize=12)

    axes[0, 0].set_title('Controls', **_title_props)
    axes[0, 1].set_title('Positions', **_title_props)
    axes[0, 2].set_title('Velocities', **_title_props)
    axes[0, 3].set_title('Attitude', **_title_props)
    axes[0, 4].set_title('Rates', **_title_props)
    axes[3, 1].set_title('Course', **_title_props)
    axes[3, 2].set_title('Angle of attack', **_title_props)
    axes[3, 3].set_title('Sideslip', **_title_props)
    axes[3, 4].set_title('Running cost', **_title_props)

    if any([label is not None for label in sim_labels]):
        make_legend(axes[0, 0], fontsize=12, loc='lower right')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'time_series.pdf'))
        plt.close()

    fig_3d = plt.figure(**fig_kwargs_3d)
    ax = fig_3d.add_subplot(projection='3d')
    ax.set_title('Flight plath', **_title_props)

    for state_traj, sim, label in zip(states, sims, sim_labels):
        pos = get_positions(sim['t'], state_traj)
        ax.plot(pos[1], pos[0], -state_traj.pd, label=label)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xylim = [np.minimum(xlim[0], ylim[0]), np.maximum(xlim[1], ylim[1])]
    ax.set_xlim(xylim)
    ax.set_ylim(xylim)

    ax.set_xlabel(_pos_labels_detail[1], fontsize=12)
    ax.set_ylabel(_pos_labels_detail[0], fontsize=12)
    ax.set_zlabel(_pos_labels_detail[2], fontsize=12)

    if any([label is not None for label in sim_labels]):
        make_legend(ax, fontsize=12)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'flight_path_3d.pdf'))
        plt.close()

    if save_dir is None:
        return fig_ts, fig_3d


def get_positions(t, states):
    d_pos = states.body_to_inertial(states.velocity)
    pos_fun = CubicSpline(t, d_pos, axis=1).antiderivative()
    return pos_fun(t)
