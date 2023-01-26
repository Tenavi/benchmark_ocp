import numpy as np
from scipy.optimize import _numdiff

def saturate(u, min=None, max=None):
    """
    Hard saturation of controls between given bounds.

    Parameters
    ----------
    u : (n_controls, n_data) or (n_controls,) array
        Control(s) to saturate.
    min : (n_controls, 1) or (n_controls,) array, optional
        Lower control bounds.
    max : (n_controls, 1) or (n_controls,) array, optional
        Upper control bounds.

    Returns
    -------
    u : array with same shape as input
        Control(s) `u` saturated between `min` and `max`.
    """
    if np.ndim(u) < 2:
        if hasattr(min, "flatten"):
            min = min.flatten()
        if hasattr(max, "flatten"):
            max = max.flatten()

    if min is None and max is None:
        return u

    return np.clip(u, min, max)

# ------------------------------------------------------------------------------

def check_int_input(n, argname, min=None):
    """
    Convert an input to an int, raising errors if this is not possible without
    likely loss of information or if the int is less than a specified minimum.

    Parameters
    ----------
    n : array-like
        Input to check. Raises `TypeError` if `n` is not an `int` or `ndarray`.
    argname : str
        How to refer to the argument `n` in error messages.
    min : int, optional
        Minimum value which `n` should take. Raises `ValueError` if `n < min`.

    Returns
    -------
    n : int
        Input `n` converted to an `int`, if possible.
    """
    if not isinstance(argname, str):
        raise TypeError("argname must be a str")
    if min is not None:
        min = check_int_input(min, "min")

    bad_type = False

    if np.size(n) != 1:
        bad_type = True
    else:
        if isinstance(n, list):
            n = np.asarray(n)

        if hasattr(n, "dtype") and "int" in str(n.dtype):
            n = int(n)
        elif not isinstance(n, int):
            bad_type = True

    if bad_type:
        raise TypeError("%s must be an int" % argname)

    if min is not None and n < min:
        raise ValueError("%s must be greater than or equal to %d" % (argname, min))

    return n

def resize_vector(array, n_rows):
    """
    Reshapes or resizes an array-like to a 2d array with one column and a
    specified number of rows.

    Parameters
    ----------
    array : array-like
        Array to reshape or resize into shape `(n_rows,1)`.
    n_rows : {int >= 1, -1}
        Number of rows desired in `x`. If `n_rows == -1` then uses
        `n_rows = np.size(x)`.
    Returns
    -------
    reshaped_array : (n_rows, 1) array
        If `array.shape == (n_rows,1)` then returns the original `array`,
        otherwise a copy is returned.
    """
    n_rows = check_int_input(n_rows, "n_rows")
    if n_rows == -1:
        n_rows = np.size(array)
    elif n_rows <= 0:
        raise ValueError("n_rows must be a positive int or -1")

    if hasattr(array,"shape") and array.shape == (n_rows,1):
        return array

    array = np.reshape(array, (-1,1))
    if array.shape[0] == n_rows:
        return array
    elif array.shape[0] == 1:
        return np.tile(array, (n_rows,1))
    else:
        raise ValueError("The size of array is not compatible with the desired shape (n_rows,1)")

# ------------------------------------------------------------------------------

def approx_derivative(
        fun, x0, method="3-point", rel_step=None, abs_step=None, f0=None,
        args=(), kwargs={}
    ):
    """
    Compute (batched) finite difference approximation of the derivatives of an
    array-valued function. Modified from
    `scipy.optimize._numdiff.approx_derivative` to allow for array-valued
    functions evaluated at multiple inputs.

    If a function maps from R^n to R^m, its derivatives form m-by-n matrix
    called the Jacobian, where an element (i, j) is a partial derivative of
    f[i] with respect to x[j].

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is an ndarray of shape (n,) or (n, n_points). It
        must return a float or an n-D array_like of shape (n_points,),
        (m_1, ..., m_l), or (m_1, ..., m_l, n_points), depending on the shape of
        the input.
    x0 : array_like of shape (n,) or (n, n_points)
        Point(s) at which to estimate the derivatives.
    method : {"3-point", "2-point", "cs"}, optional
        Finite difference method to use:
            - "2-point" - use the first order accuracy forward or backward
                          difference.
            - "3-point" - use central difference
            - "cs" - use a complex-step finite difference scheme. This assumes
                     that the user function is real-valued and can be
                     analytically continued to the complex plane. Otherwise,
                     produces bogus results.
    rel_step : None or array_like, optional
        Relative step size to use. If None (default) the absolute step size is
        computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
        `rel_step` being selected automatically, see Notes. Otherwise
        ``h = rel_step * sign(x0) * abs(x0)``. For ``method="3-point"`` the
        sign of `h` is ignored.
    abs_step : array_like, optional
        Absolute step size to use. For ``method="3-point"`` the sign of
        `abs_step` is ignored. By default relative steps are used, only if
        ``abs_step is not None`` are absolute steps used.
    f0 : None or array_like, optional
        If not None it is assumed to be equal to ``fun(x0)``, in this case
        the ``fun(x0)`` is not called. Default is None.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------
    dfdx : (n,), (n_points,), (m_1, ..., m_l, n) or (m_1, ..., m_l, n, n_points)
        array, depending on the sizes of x0 and fun(x0)
        Finite difference approximation of the Jacobian matrix or matrices.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point dtype of `x0` or `fun(x0)`,
    ``np.finfo(x0.dtype).eps``, s=2 for "2-point" method and s=3 for "3-point"
    method. Such relative step approximately minimizes a sum of truncation and
    round-off errors. Relative steps are used by default. However, absolute
    steps are used when ``abs_step is not None``. If any of the absolute or
    relative steps produces an indistinguishable difference from the original
    `x0`, ``(x0 + dx) - x0 == 0``, then an automatic step size is substituted
    for that particular entry.
    """
    if method not in ["2-point", "3-point", "cs"]:
        raise ValueError("Unknown method '%s'. " % method)

    x0 = np.atleast_1d(x0)

    flatten = [False]

    def fun_wrapped(x):
        f = fun(x, *args, **kwargs)
        if f.ndim < 1:
            flatten[0] = True
        return np.atleast_1d(f)

    if f0 is None:
        f0 = fun_wrapped(x0)
    else:
        f0 = np.atleast_1d(f0)

    # By default we use rel_step
    if abs_step is None:
        h = _numdiff._compute_absolute_step(rel_step, x0, f0, method)
    else:
        # user specifies an absolute step
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = abs_step

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
        dx = ((x0 + h) - x0)
        h_alt = (
            _numdiff._eps_for_method(x0.dtype, f0.dtype, method)
            * sign_x0 * np.maximum(1.0, np.abs(x0))
        )
        h = np.where(dx == 0, h_alt, h)

    dfdx = _dense_difference(fun_wrapped, x0, f0, h, method)

    if flatten[0]:
        return dfdx[0]
    else:
        return dfdx

def _dense_difference(fun, x0, f0, h, method):
    dfdx_T = np.empty(x0.shape[:1] + f0.shape)

    for i in range(x0.shape[0]):
        if method == "2-point":
            x = np.copy(x0)
            x[i] += h[i]
            dx = x[i] - x0[i]  # Recompute dx as exactly representable number.
            df = fun(x) - f0
        elif method == "3-point":
            x1, x2 = np.copy(x0), np.copy(x0)
            x1[i] += h[i]
            x2[i] -= h[i]
            dx = x2[i] - x1[i]
            df = fun(x2) - fun(x1)
        elif method == "cs":
            x = x0.astype("complex128")
            x[i] += h[i] * 1.j
            df = fun(x).imag
            dx = h[i]

        dfdx_T[i] = df / dx

    if x0.ndim < 2:
        return np.moveaxis(dfdx_T, 0, -1)

    return np.moveaxis(dfdx_T, 0, -2)

# ------------------------------------------------------------------------------

def find_fixed_point(OCP, controller, tol, X0=None, verbose=True):
    """
    Use root-finding to find a fixed point (equilibrium) of the closed-loop
    dynamics near the desired goal state OCP.X_bar. ALso computes the
    closed-loop Jacobian and its eigenvalues.

    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    tol : float
        Maximum value of the vector field allowed for a trajectory to be
        considered as convergence to an equilibrium
    X0 : array, optional
        Initial guess for the fixed point. If X0=None, use OCP.X_bar
    verbose : bool, default=True
        Set to True to print out the deviation of the fixed point from OCP.X_bar
        and the Jacobian eigenvalue

    Returns
    -------
    X_star : (n_states, 1) array
        Closed-loop equilibrium
    X_star_err : float
        ||X_star - OCP.X_bar||
    F_star : (n_states, 1) array
        Vector field evaluated at X_star. If successful should have F_star ~ 0
    Jac : (n_states, n_states) array
        Close-loop Jacobian at X_star
    eigs : (n_states, 1) complex array
        Eigenvalues of the closed-loop Jacobian
    max_eig : complex scalar
        Largest eigenvalue of the closed-loop Jacobian
    """
    raise NotImplementedError

    if X0 is None:
        X0 = OCP.X_bar
    X0 = np.reshape(X0, (OCP.n_states,))

    def dynamics_wrapper(X):
        U = controller.eval_U(X)
        F = OCP.dynamics(X, U)
        C = OCP.constraint_fun(X)
        if C is not None:
            F = np.concatenate((F.flatten(), C.flatten()))
        return F

    def Jacobian_wrapper(X):
        J = OCP.closed_loop_jacobian(X, controller)
        JC = OCP.constraint_jacobian(X)
        if JC is not None:
            J = np.vstack((
                J.reshape(-1,X.shape[0]), JC.reshape(-1,X.shape[0])
            ))
        return J

    sol = root(dynamics_wrapper, X0, jac=Jacobian_wrapper, method="lm")

    X_star = sol.x.reshape(-1,1)
    U_star = controller(X_star)
    F_star = OCP.dynamics(X_star, U_star).reshape(-1,1)
    Jac = OCP.closed_loop_jacobian(sol.x, controller)

    X_star_err = OCP.norm(X_star)[0]

    eigs = np.linalg.eigvals(Jac)
    idx = np.argsort(eigs.real)
    eigs = eigs[idx].reshape(-1,1)
    max_eig = np.squeeze(eigs[-1])

    # Some linearized systems always have one or more zero eigenvalues.
    # Handle this situation by taking the next largest.
    if np.abs(max_eig.real) < tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < tol**2,
                np.abs(max_eig0.real) < tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    if verbose:
        s = "||actual - desired_equilibrium|| = {norm:1.2e}"
        print(s.format(norm=X_star_err))
        if np.max(np.abs(F_star)) > tol:
            print("Dynamics f(X_star):")
            print(F_star)
        s = "Largest Jacobian eigenvalue = {real:1.2e} + j{imag:1.2e} \n"
        print(s.format(real=max_eig.real, imag=np.abs(max_eig.imag)))

    return X_star, X_star_err, F_star, Jac, eigs, max_eig
