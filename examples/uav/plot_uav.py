import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from examples.common_utilities.dynamics import quaternion_to_euler

from examples.uav.fixed_wing_dynamics.containers import VehicleState, Controls


_control_labels = [r'$\delta_t$', r'$\delta_a$ [deg]',
                   r'$\delta_e$ [deg]', r'$\delta_r$ [deg]']
_control_scales = [1.] + [180. / np.pi] * 3
_pos_labels = [r'$p_n$ [m]', r'$p_e$ [m]', r'$h - h_f$ [m]']
_vel_labels = [r'$u$ [m/s]', r'$v$ [m/s]', r'$w$ [m/s]']
_eul_labels = [r'$\phi$ [deg]', r'$\theta$ [deg]', r'$\psi - \psi_f$ [deg]']
_pqr_labels = [r'$p$ [deg/s]', r'$q$ [deg/s]', r'$r$ [deg/s]']
_chi_alpha_beta_labels = [r'$\chi - \chi_f$ [deg]', r'$\alpha$ [deg]',
                          r'$\beta$ [deg]']


def plot_closed_loop(sims, ocp, sim_labels=None, x_min=None, x_max=None,
                     subtitle=None, fig_kwargs_ts={}, fig_kwargs_3d={},
                     save_dir=None):
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
    subtitle : str, optional
        If provided, this string appears in parentheses after the first plot
        title.
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

    if subtitle is not None:
        fig_ts.suptitle(f'Closed-loop states ({subtitle})', fontsize=14)
    else:
        fig_ts.suptitle('Closed-loop state', fontsize=14)

    for i in range(4):
        for j in range(5):
            ax = axes[i, j]
            ax.set_xlim(0., t_max)
            if i == 3:
                ax.set_xlabel('$t$ [s]', fontsize=12)

    for i in range(4):
        ax = axes[i, 0]

        for sim, label in zip(sims, sim_labels):
            ax.plot(sim['t'], sim['u'][i] * _control_scales[i], label=label)

        ax.set_ylim(ocp.control_lb[i] * _control_scales[i],
                    ocp.control_ub[i] * _control_scales[i])

        ax.set_ylabel(_control_labels[i], fontsize=12)

    for state_traj, sim, label in zip(states, sims, sim_labels):
        pos = get_positions(sim['t'], state_traj)
        pos[-1] = -state_traj.pd
        eul_angles = quaternion_to_euler(state_traj.attitude, degrees=True)
        rates = np.rad2deg(state_traj.rates)

        for i in range(3):
            axes[i, 1].plot(sim['t'], pos[i], label=label)
            axes[i, 2].plot(sim['t'], state_traj.velocity[i], label=label)
            axes[i, 3].plot(sim['t'], eul_angles[::-1][i], label=label)
            axes[i, 4].plot(sim['t'], rates[i], label=label)

        ax = axes[3, 1]
        ax.plot(sim['t'], np.rad2deg(state_traj.course), label=label)
        ax = axes[3, 2]
        ax.plot(sim['t'], np.rad2deg(state_traj.airspeed[1]), label=label)
        ax = axes[3, 3]
        ax.plot(sim['t'], np.rad2deg(state_traj.airspeed[2]), label=label)

    for i in range(3):
        axes[i, 1].set_ylabel(_pos_labels[i], fontsize=12)
        axes[i, 2].set_ylabel(_vel_labels[i], fontsize=12)
        axes[i, 3].set_ylabel(_eul_labels[i], fontsize=12)
        axes[i, 4].set_ylabel(_pqr_labels[i], fontsize=12)

    for j in range(3):
        axes[3, 1 + j].set_ylabel(_chi_alpha_beta_labels[j], fontsize=12)

    if save_dir is None:
        return fig_ts#, fig_3d

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'sim{i:d}.pdf'))
        plt.close()


def get_positions(t, states):
    d_pos = states.body_to_inertial(states.velocity)
    pos_fun = CubicSpline(t, d_pos, axis=1).antiderivative()
    return pos_fun(t)
