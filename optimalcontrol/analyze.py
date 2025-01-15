import numpy as np

from .simulate import integrate_to_converge
from .utilities import closed_loop_jacobian


def linear_stability(ocp, controller, x, zero_tol=1e-08, verbose=True):
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
    verbose : bool, default=True
        If `verbose=True` (default), then print out the largest eigenvalue.

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

    if verbose:
        print(f"Largest non-zero Jacobian eigenvalue = "
              f"{max_eig.real:.4g} + j{np.abs(max_eig.imag):.4g}")

    return jac, eigs, max_eig


def find_equilibrium(ocp, controller, x0, t_int, t_max, **kwargs):
    r"""
    Finds an equilibria of the closed-loop dynamics, $dx/dt = f(x, u(x))$, near
    a given point `x0`.

    This is accomplished by integrating both forwards and backwards in time
    using `simulate.integrate_to_converge` until a maximum time horizon or
    dynamic equilibrium, $f(x, u(x)) = 0$, is reached. Integrating both
    directions allows both stable and unstable equilibria to be found. The
    integration `status` of each integration is also returned to inform the
    selection of the appropriate point.

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
    x : (`ocp.n_states, 2`) array
        States found by integrating forward (`x[:, 0]`) and backward (`x[:, 1]`)
        in time. If `status[i] == 0` then that point is an equilibrium.
    status : (2,) int array
        Reasons for integration termination. `status[0]` contains the forward
        integration status corresponding to `x[:, 0]`, and `status[1]` contains
        the backward integration status corresponding to `x[:, 1]`.

            * -1: Integration step failed.
            *  0: The system reached a steady state as determined by `ftol`.
            *  1: A termination event occurred.
            *  2: `t[-1]` exceeded `t_max`.
            *  3: Both forward and backward integration converged to equilibria,
                but this equilibrium was further from `x0`.
    """
    t_int = np.abs(t_int)
    t_max = np.abs(t_max)

    # Setup array to store forward and backward integration solutions
    x = np.tile(np.reshape(x0, (ocp.n_states, 1)), (1, 2))

    status = np.empty((2,), dtype=int)

    # Forward and backwards integration
    for i, sign in enumerate([1., -1.]):
        _, x_sol, status[i] = integrate_to_converge(ocp, controller, x0,
                                                    t_int * sign, t_max * sign,
                                                    **kwargs)
        x[:, i] = x_sol[:, -1]

    # If both forward and backwards integrations converged to an equilibrium,
    # check which point is closer to the start
    if np.all(status == 0):
        dists = ocp.distances(x, x0).reshape(2)
        if dists[0] <= dists[1]:
            status[1] = 3
        else:
            status[0] = 3

    return x, status
