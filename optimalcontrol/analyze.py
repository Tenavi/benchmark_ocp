"""
TODO: write module documentation
"""

import warnings

import numpy as np

from .simulate import integrate_to_converge
from .utilities import closed_loop_jacobian


def linear_stability(ocp, controller, x, zero_tol=1e-08):
    r"""
    Find the eigenvalues and the maximum non-zero eigenvalue of the closed-loop
    Jacobian matrix, $Df/Dx = df/dx + df/du \cdot du/dx$.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        The dynamical system to analyze.
    controller : `Controller`
        The feedback controller closing the loop.
    x : (`ocp.n_states`,) array
        Equilibrium point to analyze.
    zero_tol : float, default=1e-08
        Tolerance for considering an eigenvalue to have zero real part, i.e.
        eigenvalues with `abs(real(eigs)) < zero_tol` are considered to be zero.

    Returns
    -------
    jac : (`ocp.n_states`, `ocp.n_states`) array
        Closed-loop Jacobian at `x`.
    eigs : (n_states,) complex array
        Eigenvalues of `jac`, ordered from largest to largest real part.
    max_eig : complex scalar
        Largest non-zero eigenvalue of `jac`.
    """
    x = np.reshape(x, (ocp.n_states,))
    jac = closed_loop_jacobian(x, ocp.jac, controller)

    eigs = np.linalg.eigvals(jac)
    eigs = eigs[np.argsort(eigs.real)]
    i = eigs.shape[0] - 1
    max_eig = eigs[i]

    while np.isclose(max_eig.real, 0., atol=zero_tol) and i >= 1:
        i -= 1
        max_eig = eigs[i]

    print(f"Largest non-zero Jacobian eigenvalue = "
          f"{max_eig.real:1.2e} + j{np.abs(max_eig.imag):1.2e}")

    return jac, eigs, max_eig


def find_equilibrium(ocp, controller, x0, t_int, t_max, **kwargs):
    r"""
    Finds an equilibrium of the closed-loop dynamics, $dx/dt = f(x, u(x))$, near
    a given point `x0`.

    This is accomplished by integrating both forwards and backwards in time
    using `simulate.integrate_to_converge` until a maximum time horizon or
    dynamic equilibrium, $f(x, u(x)) = 0$, is reached. Integrating both
    directions allows both stable and unstable equilibria to be found. If both
    integrations converged to an equilibrium, returns the point which is closest
    to the initial guess. If neither integration converged, raises a
    `RuntimeWarning` and returns the last points evaluated in both directions
    and the integration `status` returned by `integrate_to_converge` for each.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `dynamics` and `jac` methods.
    controller : `Controller`
        An instance of a `Controller` subclass implementing `__call__` and `jac`
        methods.
    x0 : (`ocp.n_states`,) array
        Initial guess for the equilibrium point.
    t_int : float
        Time interval to step integration over (see `integrate_to_converge`).
    t_max : float
        Maximum time allowed for integration.
    **kwargs : dict
        Keyword arguments to pass to `integrate_to_converge`.

    Returns
    -------
    x : (`ocp.n_states`,) array
        Closed-loop equilibrium.
    f : (`ocp.n_states`,) array
        Vector field evaluated at `x, controller(x)`. If root-finding was
        successful should have `f` approximately zero.
    """
    t_int, t_max = np.abs(t_int), np.abs(t_max)

    # Setup array to store forward and backward integration solutions
    x = np.tile(np.reshape(x0, (ocp.n_states, 1)), (1, 2))

    status = np.empty((2,), dtype=int)

    # Forward and backwards integration
    for i, sign in enumerate([1, -1]):
        t_sol, x_sol, status[i] = integrate_to_converge(ocp, controller, x0,
                                                        t_int * sign,
                                                        t_max * sign, **kwargs)
        x[:, i] = x_sol[:, -1]

    # If both forward and backwards integrations converged to an equilibrium,
    # check which point is closer to the start
    if np.all(status == 0):
        dists = ocp.distances(x, x0)
        status[np.argmax(dists)] = 3
    elif np.all(status != 0):
        warnings.warn("No equilibrium was found", RuntimeWarning)
        return x, status

    return x[:, status == 0].reshape(-1)
