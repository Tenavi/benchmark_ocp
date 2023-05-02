import numpy as np
from scipy.optimize import root

from .utilities import closed_loop_jacobian


def linear_stability(jac):
    r"""
    Find the eigenvalues and the maximum non-zero eigenvalue of the closed-loop
    Jacobian matrix, $Df/Dx = df/dx + df/du \cdot du/dx$.

    Parameters
    ----------
    jac : (n_states, n_states) array
        Closed-loop Jacobian matrix.

    Returns
    -------
    eigs : (n_states,) complex array
        Eigenvalues of `jac`, ordered from smallest to larges.
    max_eig : complex scalar
        Largest non-zero eigenvalue of `jac`.
    """
    eigs = np.linalg.eigvals(jac)
    eigs = eigs[np.argsort(eigs.real)]
    i = eigs.shape[0] - 1
    max_eig = eigs[i]

    while np.isclose(max_eig.real, 0., atol=1e-12) and i >= 1:
        i -= 1
        max_eig = eigs[i]

    print(f"Largest non-zero Jacobian eigenvalue = "
          f"{max_eig.real:1.2e} + j{np.abs(max_eig.imag):1.2e}")

    return eigs, max_eig


def find_equilibrium(ocp, controller, x0, **root_opts):
    r"""
    Uses root-finding (`scipy.optimize.root`) to find an equilibrium of the
    closed-loop dynamics, $dx/dt = f(x,u(x))$, near a given point `x0`.

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
    **root_opts : dict
        Keyword arguments to pass to `scipy.optimize.root`.

    Returns
    -------
    x : (`ocp.n_states`,) array
        Closed-loop equilibrium.
    f : (`ocp.n_states`) array
        Vector field evaluated at `x, controller(x)`. If root-finding was
        successful should have `f` approximately zero.
    """
    x0 = np.reshape(x0, (ocp.n_states,))

    c0 = ocp.constraint_fun(x0)
    if c0 is not None:
        raise NotImplementedError("find_equilibrium cannot yet handle state "
                                  "constraints")

    def dynamics_wrapper(x):
        u = controller(x)
        return ocp.dynamics(x, u)

    def jac_wrapper(x):
        return closed_loop_jacobian(x, ocp.jac, controller)

    sol = root(dynamics_wrapper, x0, jac=jac_wrapper, **root_opts)

    f = dynamics_wrapper(sol.x)

    print("Equilibrium point x:")
    print(sol.x.reshape(-1, 1))

    return sol.x, f
