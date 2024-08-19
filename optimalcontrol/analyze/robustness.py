import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def disk_margins(ocp, controller, x, w=np.logspace(-1., 2., 300), skew=0.,
                 bound='tight', plot=False, **minimize_kwargs):
    """
    Evaluate multi-loop disk margins at the input channel for the closed-loop
    system linearized about an equilibrium point.

    Disk margins generalize classical gain and phase margins to MIMO systems,
    accounting for simultaneous gain and phase variations in multiple channels.
    Disk margins are computed by maximizing an upper bound on the structured
    singular value of the closed-loop system model for each frequency in a given
    range. This provides a frequency-dependent measure of robustness.

    We also extract the critical frequency where the bound is smallest; the
    reported `disk_margin` is the margin at this frequency. We can also
    interpret the disk margin in terms of simultaneous classical gain and phase
    margins. In particular, the input-channel of the linearized system can
    tolerate simultaneous gain and phase perturbations up to the reported
    `gain_margin` and `phase_margin`.

    For a detailed explanation see refs. [1, 2].

    Disk margin computation for the output channel in a state feedback control
    system are still under development.

    ##### References

    1. P. Seiler, A. Packard, and P. Gahinet, An introduction to disk margins
        [Lecture notes], IEEE Control Systems Magazine, 40 (2020), pp. 78-95.
        https://doi.org/10.1109/MCS.2020.3005277
    2. E. Lavretsky and K. A. Wise, Robust and Adaptive Control with Aerospace
        Applications (2nd ed), Springer, 2024.
        https://doi.org/10.1007/978-3-031-38314-4

    Parameters
    ----------
    ocp : OptimalControlProblem
        Control problem implementing dynamics to analyze.
    controller : Controller
        Feedback control law to analyze.
    x : (`n_states`,) array
        Equilibrium point at which to linearize the dynamics.
    w : (`n_frequencies`,) array, default=`np.logspace(-1, 2, 100)`
        Frequencies (rad/s) at which to analyze the system.
    skew : float, default=0
        Parameter "sigma" in disk margin computation (see ref. [1]). Common
        values include `skew=0` (default), `skew=1` ("S-vector margins", see
        ref. [2]), and `skew=-1` ("T-vector margins", see ref. [2]).
    bound : {'tight', 'relaxed'}, default='tight'
        How to compute the upper bound on the structured singular value, mu. The
        default 'tight' bound minimizes the maximum singular value of
        `D @ M @ inv(D)` where `D` is a diagonal matrix. The 'relaxed' bound
        minimizes the Frobenius norm of this quantity, which is greater than the
        singular value. The 'relaxed' bound produces a slightly more
        conservative disk margin, but it is faster to compute and so can be
        useful if many computations need to be performed for a large-scale
        system.
    plot : bool, default=False
        If `plot=True`, produces a figure with plots of the maximum singular
        values of the loop gain transfer matrix (L), sensitivity (S),
        cosensitivity (T), as well as the frequency-dependent gain and phase
        margins. See ref. [1] on how to interpret the former three plots.
    minimize_kwargs : dict, default={'method': 'SLSQP'}
        Keyword arguments to pass to `scipy.optimize.minimize` when solving for
        the upper bound on the structured singular value.

    Returns
    -------
    margins : dict
        Results of the margin computation with key-value pairs

            * 'disk_margin': minimum disk margin over all frequencies `w`
                (given in standard values, not dB)
            * 'gain_margin': gain margins based on the minimum disk margin
                (given in standard values, not dB)
            * 'phase_margin': phase margins based on the minimum disk margin
                (given in degrees)
            * 'critical_frequency': frequency at which the disk margin is
                smallest.
    """
    x = np.reshape(x, (ocp.n_states,))
    u = controller(x)

    if np.size(skew) != 1:
        raise ValueError("skew must be a scalar")

    w = np.sort(w)

    if w[0] <= 0. or np.any(np.iscomplex(w)):
        raise ValueError("w must be a vector of positive real numbers")

    minimize_kwargs = {'method': 'SLSQP', **minimize_kwargs,
                       'skew': skew, 'bound': bound}

    # Generate analysis matrices
    A, B = ocp.jac(x, u)
    K = controller.jac(x, u)
    L, _ = _loop_gain(w, A, B, K)

    dm, S, M = _disk_margins_general(w, L, **minimize_kwargs)
    margins, g_min, g_max, phi = _process_margins(w, dm, skew)
    margins['crossover_frequency'], sigma_L = _gain_crossover(w, L)

    if plot:
        _plot_margins(w, skew, g_min, g_max, phi, sigma_L, S, M)

    return margins


def _disk_margins_io(ocp, controller, x, w=np.logspace(-1., 2., 100), skew=0.,
                     bound='tight', plot=False, **minimize_kwargs):
    warnings.warn(UserWarning("Output margins may not behave as expected, use "
                              "with caution."))

    x = np.reshape(x, (ocp.n_states,))
    u = controller(x)

    if np.size(skew) != 1:
        raise ValueError("skew must be a scalar")

    w = np.sort(w)

    if w[0] <= 0. or np.any(np.iscomplex(w)):
        raise ValueError("w must be a vector of positive real numbers")

    minimize_kwargs = {'method': 'SLSQP', **minimize_kwargs,
                       'skew': skew, 'bound': bound}

    # Generate analysis matrices
    A, B = ocp.jac(x, u)
    K = controller.jac(x, u)

    L_u, L_y = _loop_gain(w, A, B, K)

    dm_u, S_u, M_u = _disk_margins_general(w, L_u, **minimize_kwargs)
    margins_u, g_min_u, g_max_u, phi_u = _process_margins(w, dm_u, skew)
    margins_u['crossover_frequency'], sigma_L_u = _gain_crossover(w, L_u)

    dm_y, S_y, M_y = _disk_margins_general(w, L_y, **minimize_kwargs)
    margins_y, g_min_y, g_max_y, phi_y = _process_margins(w, dm_y, skew)
    margins_y['crossover_frequency'], sigma_L_y = _gain_crossover(w, L_y)

    if plot:
        _plot_margins_io(w, skew,
                        [g_min_u, g_min_y], [g_max_u, g_max_y], [phi_u, phi_y],
                        [sigma_L_u, sigma_L_y], [S_u, S_y], [M_u, M_y])

    return margins_u, margins_y


def _disk_margins_general(w, L, skew=0., bound='tight', **minimize_kwargs):
    if bound == 'tight':
        minimize_obj = _mu_svd
    elif bound == 'relaxed':
        minimize_obj = _mu_fro
    else:
        raise ValueError(f"bound = {bound} is an invalid option; must be one "
                         f"of 'tight' or 'relaxed'")

    S = _sensitivity(L)
    M = _system_matrix(S, skew)

    mu_ssv = np.empty_like(w)
    d = np.ones(M.shape[:2])

    for i, Mi in enumerate(M):
        mu_ssv[i], d[i] = _optimize_scaling(Mi, f=minimize_obj, d0=d[i - 1],
                                            **minimize_kwargs)

    # Recompute objective value if using 'relaxed' bound
    if bound == 'relaxed':
        mu_ssv = _mu_svd(d, M)

    disk_margin = 1. / mu_ssv

    return disk_margin, S, M


def _process_margins(w, disk_margin, skew):
    # Find smallest margins
    idx_w_c = disk_margin.argmin()
    alpha = disk_margin[idx_w_c]
    w_c = w[idx_w_c]

    g_min, g_max, phi = _classical_margins(disk_margin, skew)

    margins = {'disk_margin': alpha,
               'gain_margin': np.array([g_min[idx_w_c], g_max[idx_w_c]]),
               'phase_margin': np.array([-phi[idx_w_c], phi[idx_w_c]]),
               'critical_frequency': w_c}

    return margins, g_min, g_max, phi


def _loop_gain(w, A, B, K):
    n_x, n_u = B.shape
    jw_I = np.einsum('b,ij->bij', 1j * w, np.eye(n_x))
    L = np.linalg.solve(jw_I - A, B.reshape(1, n_x, n_u))
    L_u = np.einsum('ij,bjk->bik', -K, L)
    L_y = np.einsum('bij,jk->bik', L, -K)
    return L_u, L_y


def _sensitivity(L):
    Sinv = L + np.eye(L.shape[1])[None, ...]
    return np.linalg.inv(Sinv)


def _system_matrix(S, skew=0.):
    return S + (skew - 1.) / 2. * np.eye(S.shape[1])[None, ...]


def _scale_matrix(d, M):
    """Batch computes the matrix product diag(d) @ M @ inv(diag(d))."""
    M_D_inv = np.einsum('...ij,...j->...ij', M, 1. / d)
    return np.einsum('...ij,...i->...ij', M_D_inv, d)


def _mu_svd(d, M):
    sigma = np.linalg.svd(_scale_matrix(d, M), compute_uv=False)
    return sigma.max(axis=-1)


def _mu_fro(d, M):
    return np.linalg.norm(_scale_matrix(d, M), 'fro', axis=(-2, -1))


def _optimize_scaling(M, f=_mu_svd, d0=None, **kwargs):
    n = M.shape[1]

    # Initialize variables
    if d0 is None:
        d0 = np.ones(n)

    def expand_d(d):
        return np.concatenate(([1.], np.exp(d)))

    if n == 1:
        return f(d0, M), d0

    sol = minimize(lambda d: f(expand_d(d), M), np.log(d0[1:]), **kwargs)

    return sol.fun, expand_d(sol.x)


def _classical_margins(alpha, skew=0.):
    alpha_sigma = np.outer([1. - skew, 1. + skew], alpha)

    g_min = (2. - alpha_sigma[0]) / (2. + alpha_sigma[1])
    g_max = (2. + alpha_sigma[0]) / (2. - alpha_sigma[1])

    g_min = np.minimum(g_min, 1.)
    g_max = np.maximum(np.abs(g_max), 1.)

    g_min_pos = np.maximum(g_min, 0.)

    phi = np.arccos((1 + g_min_pos * g_max) / (g_min_pos + g_max))

    return g_min, g_max, np.rad2deg(phi)


def _gain_crossover(w, L):
    sigma_L = np.linalg.svd(L, compute_uv=False).max(axis=1)
    idx_w_g = np.abs(1. - sigma_L).argmin()
    return w[idx_w_g], sigma_L


def _decibels(x):
    return 20. * np.log10(x)


def _plot_margins(w, skew, g_min, g_max, phi, sigma_L, S, M):
    fig, ax = plt.subplots(4, figsize=(8, 8), constrained_layout=True)

    sigma_S = np.linalg.svd(S, compute_uv=False).max(axis=1)

    if skew == -1.:
        T = M
    else:
        T = _system_matrix(S, -1.)
    sigma_T = np.linalg.svd(T, compute_uv=False).max(axis=1)

    g_min_plot = g_max.copy()
    idx = g_min > 0.
    g_min_plot[idx] = np.minimum(1. / g_min[idx], g_min_plot[idx])

    ax[0].plot(w, _decibels(sigma_L))
    ax[1].plot(w, _decibels(sigma_S), label=r'sensitivity $\bar \sigma (S_u)$')
    ax[1].plot(w, _decibels(sigma_T),
               label=r'cosensitivity $\bar \sigma (T_u)$')
    ax[2].plot(w, _decibels(g_min_plot))
    ax[3].plot(w, phi)

    ax[1].legend(fontsize=12)

    ax[0].set_title(r"Loop gain $\bar \sigma (L_u)$", fontsize=12)
    ax[1].set_title("Sensitivities", fontsize=12)
    ax[2].set_title("Gain margin", fontsize=12)
    ax[3].set_title("Phase margin", fontsize=12)

    for a in ax[:-1]:
        a.set_ylabel("magnitude (dB)", fontsize=10)

    ax[-1].set_ylabel(r"phase ($^\circ$)", fontsize=10)

    for a in ax:
        a.set_xscale('log')
        a.set_xlim(w[0], w[-1])
        a.set_xlabel(r"frequency $\omega$ (rad/s)", fontsize=10)
        a.grid()

    plt.show()


def _plot_margins_io(w, skew, g_min, g_max, phi, sigma_L, S, M):
    fig, ax = plt.subplots(6, figsize=(8, 8), constrained_layout=True)

    for i, ss in enumerate(['u', 'y']):
        sigma_S = np.linalg.svd(S, compute_uv=False).max(axis=1)

        if skew == -1.:
            T = M
        else:
            T = _system_matrix(S, -1.)
        sigma_T = np.linalg.svd(T, compute_uv=False).max(axis=1)

        g_min_plot = g_max.copy()
        idx = g_min > 0.
        g_min_plot[idx] = np.minimum(1. / g_min[idx], g_min_plot[idx])

        ax[0].plot(w, _decibels(sigma_L), label=rf'$\bar \sigma(L_{ss})$')
        ax[1].plot(w, _decibels(sigma_S), label=rf'$\bar \sigma(S_{ss})$')
        ax[2].plot(w, _decibels(sigma_T), label=rf'$\bar \sigma(T_{ss})$')
        ax[3].plot(w, _decibels(g_min_plot),
                   label=r'$\gamma_' + '{min, ' + ss + '}$')
        ax[4].plot(w, phi, label=r'$\phi_' + '{min, ' + ss + '}$')

    ax[0].set_title("Loop gain (L)", fontsize=12)
    ax[1].set_title("Sensitivities (S)", fontsize=12)
    ax[2].set_title("Cosensitivities (T)", fontsize=12)
    ax[3].set_title("Gain margins", fontsize=12)
    ax[4].set_title("Phase margins", fontsize=12)

    for a in ax[:-1]:
        a.set_ylabel("magnitude (dB)", fontsize=10)

    ax[-1].set_ylabel(r"phase ($^\circ$)", fontsize=10)

    for a in ax:
        a.legend(fontsize=12)
        a.set_xscale('log')
        a.set_xlim(w[0], w[-1])
        a.set_xlabel(r"frequency $\omega$ (rad/s)", fontsize=10)
        a.grid()

    plt.show()
