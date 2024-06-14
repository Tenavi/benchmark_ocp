import numpy as np
import pandas as pd
from scipy.optimize import _numdiff


def saturate(u, lb=None, ub=None):
    """
    Hard saturation of controls between given bounds.

    Parameters
    ----------
    u : (n_controls, n_data) or (n_controls,) array
        Control(s) to saturate.
    lb : (n_controls, 1) or (n_controls,) array, optional
        Lower control bounds.
    ub : (n_controls, 1) or (n_controls,) array, optional
        Upper control bounds.

    Returns
    -------
    u : array with same shape as `u`
        Control(s) `u` saturated between `lb` and `ub`.
    """
    if lb is None and ub is None:
        return u

    lb, ub = _reshape_bounds(np.ndim(u), lb, ub)

    return np.clip(u, lb, ub)


def find_saturated(u, lb=None, ub=None):
    """
    Find indices where controls are saturated between given bounds.

    Parameters
    ----------
    u : (n_controls, n_data) or (n_controls,) array
        Control(s) arranged by dimension, time.
    lb : (n_controls, 1) or (n_controls,) array, optional
        Lower control bounds.
    ub : (n_controls, 1) or (n_controls,) array, optional
        Upper control bounds.

    Returns
    -------
    sat_idx : boolean array with same shape as `u`
        `sat_idx[i, j] = True` if `u[i, j] <= lb[i]` or `u[i, j] >= ub[i]`. If
        `lb` or `ub` is `None` then these are ignored.
    """
    lb, ub = _reshape_bounds(np.ndim(u), lb, ub)

    if lb is not None and ub is not None:
        return np.any([ub <= u, u <= lb], axis=0)
    elif ub is not None:
        return ub <= u
    elif lb is not None:
        return u <= lb
    else:
        return np.full(np.shape(u), False)


def _reshape_bounds(ndim, lb=None, ub=None):
    bound_shape = (-1,) + (1,) * (ndim - 1)

    if lb is not None and np.ndim(lb) != ndim:
        lb = np.reshape(lb, bound_shape)
    if ub is not None and np.ndim(ub) != ndim:
        ub = np.reshape(ub, bound_shape)

    return lb, ub


def check_int_input(n, argname, low=None):
    """
    Convert an input to an int, raising errors if this is not possible without
    likely loss of information or if the int is less than a specified minimum.

    Parameters
    ----------
    n : array_like, size 1
        Input to check.
    argname : str
        How to refer to the argument `n` in error messages.
    low : int, optional
        Minimum value which `n` should take.

    Raises
    ------
    TypeError
        If `n` is not an int or array_like of size 1.
    ValueError
        If `n < low`.

    Returns
    -------
    n : int
        Input `n` converted to an int, if possible.
    """
    if not isinstance(argname, str):
        raise TypeError("argname must be a str")
    if low is not None:
        low = check_int_input(low, 'low')

    try:
        n = np.squeeze(n).astype(np.int64, casting='safe')
        n = int(n)
    except TypeError:
        raise TypeError(f"{argname} must be an int")

    if low is not None and n < low:
        raise ValueError(f"{argname} must be greater than or equal to {low:d}")

    return n


def resize_vector(array, n_rows):
    """
    Reshapes or resizes an array_like to a 2d array with one column and a
    specified number of rows.

    Parameters
    ----------
    array : array_like
        Array to reshape or resize into shape `(n_rows, 1)`.
    n_rows : int
        Number of rows desired in `x`. Can be any positive int or -1. If
        `n_rows == -1` then uses `n_rows = np.size(array)`.

    Returns
    -------
    reshaped_array : (n_rows, 1) array
        If `array.shape == (n_rows, 1)` then returns the original `array`,
        otherwise a copy is returned.
    """
    n_rows = check_int_input(n_rows, "n_rows")
    if n_rows == -1:
        n_rows = np.size(array)
    elif n_rows <= 0:
        raise ValueError("n_rows must be a positive int or -1")

    if hasattr(array, 'shape') and array.shape == (n_rows, 1):
        return array

    array = np.reshape(array, (-1, 1))
    if array.shape[0] == n_rows:
        return array
    elif array.shape[0] == 1:
        return np.tile(array, (n_rows, 1))
    else:
        raise ValueError("The size of array is not compatible with the desired "
                         "shape (n_rows, 1)")


def approx_derivative(fun, x0, method="3-point", rel_step=None, abs_step=None,
                      f0=None, args=(), kwargs={}):
    """
    Compute (batched) finite difference approximation of the derivatives of an
    array-valued function. Modified from
    `scipy.optimize._numdiff.approx_derivative` to allow for array-valued
    functions evaluated at multiple inputs.

    If a function maps from $R^n$ to $R^m$, its derivatives form m-by-n matrix
    called the Jacobian, where an element `[i, j]` is a partial derivative of
    `f[i]` with respect to `x[j]`.

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument `x`
        passed to this function is an ndarray of shape `(n,)` or
        `(n, n_points)`. It must return a float or an nd array_like of shape
        `(n_points,)`, `(m_1, ..., m_l)`, or `(m_1, ..., m_l, n_points)`,
        depending on the shape of the input.
    x0 : (n,) or (n, n_points) array
        Point(s) at which to estimate the derivatives.
    method : {"3-point", "2-point", "cs"}, optional
        Finite difference method to use:

            * "2-point" - use the first order accuracy forward or backward
                          difference.
            * "3-point" - use central difference
            * "cs" - use a complex-step finite difference scheme. This assumes
                     that the user function is real-valued and can be
                     analytically continued to the complex plane. Otherwise,
                     produces bogus results.
    rel_step : array_like, optional
        Relative step size to use. If `None` (default) the absolute step size is
        computed as `h = rel_step * sign(x0) * max(1, abs(x0))`, with
        `rel_step` being selected automatically, see Notes. Otherwise
        `h = rel_step * sign(x0) * abs(x0)`. For `method="3-point"` the sign of
        `h` is ignored.
    abs_step : array_like, optional
        Absolute step size to use. For `method="3-point"` the sign of `abs_step`
        is ignored. By default relative steps are used; only if
        `abs_step is not None` are absolute steps used.
    f0 : array_like, optional
        If not `None` it is assumed to be equal to `fun(x0)`, in this case
        `fun(x0)` is not called.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`. Both empty by default.
        The calling signature is `fun(x, *args, **kwargs)`.

    Returns
    -------
    dfdx : (n,), (n_points,), (m_1, ..., m_l, n), or \
            (m_1, ..., m_l, n, n_points) array
        Finite difference approximation of the Jacobian matrix or matrices. The
        shape of `dfdx` depends on the sizes of `x0` and `fun(x0)`.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point `dtype` of `x0` or `fun(x0)`,
    `np.finfo(x0.dtype).eps`, s=2 for "2-point" method and s=3 for "3-point"
    method. Such relative step approximately minimizes a sum of truncation and
    round-off errors. Relative steps are used by default. However, absolute
    steps are used when `abs_step is not None`. If any of the absolute or
    relative steps produces an indistinguishable difference from the original
    `x0`, `(x0 + dx) - x0 == 0`, then an automatic step size is substituted for
    that particular entry.
    """
    if method not in ["2-point", "3-point", "cs"]:
        raise ValueError(f"Unknown method '{method}'. ")

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
        h_alt = (_numdiff._eps_for_method(x0.dtype, f0.dtype, method)
                 * sign_x0 * np.maximum(1.0, np.abs(x0)))
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
        else:
            raise ValueError(f"Unknown method '{method}'. ")

        dfdx_T[i] = df / dx

    if x0.ndim < 2:
        return np.moveaxis(dfdx_T, 0, -1)

    return np.moveaxis(dfdx_T, 0, -2)


def closed_loop_jacobian(x, open_loop_jac, controller):
    r"""
    Evaluate the Jacobian of closed-loop dynamics,
    $Df/Dx = df/dx + df/du \cdot du/dx$, where $f = f(x,u(x))$ is a vector field
    with partial derivatives $df/dx$ and $df/du$.

    Parameters
    ----------
    x : (n_states,) or (n_states, n_points) array
        State(s) arranged by (dimension, time).
    open_loop_jac : callable
        Function defining the open-loop partial derivatives $df/dx$ and $df/du$.
        See `OptimalControlProblem.jac`.
    controller : `Controller`
        `Controller` instance implementing `__call__` and `jac`.

    Returns
    -------
    jac : (n_states, n_states) or (n_states, n_states, n_points) array
        Closed-loop Jacobian(s), with `jac[i, j]` equal to the partial
        derivative of `f[i]` with respect to `x[j]`.
    """
    u = controller(x)
    dfdx, dfdu = open_loop_jac(x, u)
    dudx = controller.jac(x, u0=u)
    return dfdx + np.einsum('ij...,jk...->ik...', dfdu, dudx)


def pack_dataframe(t, x, u, p=None, v=None):
    """
    Collect `numpy` arrays into a `DataFrame` which is convenient for saving as
    a .csv file.

    Parameters
    ----------
    t : (n_data,) array
        Time values of each data point.
    x : (n_states, n_data) array
        System states at times `t`.
    u : (n_controls, n_data) array
        Control inputs at times `t`.
    p : (n_states, n_data) array, optional
        Costates/value at times `t`.
    v : (n_points,) array, optional
        Value function/cost-to-go at times `t`.

    Returns
    -------
    data : DataFrame
        `DataFrame` with `n_data` rows and columns 't', 'x1', ..., 'xn',
        'u1', ..., 'um', 'p1', ..., 'pn', and 'v'.
    """
    n_states = np.shape(x)[0]
    n_controls = np.shape(u)[0]

    t = np.reshape(t, (1, -1))
    x = np.reshape(x, (n_states, -1))
    u = np.reshape(u, (n_controls, -1))
    data = (t, x, u)
    if p is not None:
        if np.shape(x) != np.shape(p):
            raise ValueError('x and p must have the same shape.')
        p = np.reshape(p, (n_states, -1))
        data = data + (p,)
    if v is not None:
        v = np.reshape(v, (1, -1))
        data = data + (v,)

    data = np.vstack(data).T

    columns = (['t'] + ['x' + str(i + 1) for i in range(n_states)]
               + ['u' + str(i + 1) for i in range(n_controls)])
    if p is not None:
        columns += ['p' + str(i + 1) for i in range(n_states)]
    if v is not None:
        columns += ['v']

    return pd.DataFrame(data, columns=columns)


def unpack_dataframe(data):
    """
    Extract `numpy` `ndarray`s from a dict or `DataFrame` (formatted by
    `pack_dataframe`).

    Parameters
    ----------
    data : dict DataFrame
        dict or `DataFrame` with `n_data` rows and keys/columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' or 'x' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' or 'u' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p1', ..., 'pn' or 'p' : Costates $p_1(t)$, ..., $p_n(t)$,
                optional
            * 'v' : Value function/cost-to-go $v(t)$, optional.

    Returns
    -------
    t : (n_data,) array
    x : (n_states, n_data) array
    u : (n_controls, n_data) array
    p : (n_states, n_data) array or None
    v : (n_points,) array or None
    """
    if isinstance(data, pd.DataFrame):
        t = data['t'].to_numpy()
        x = data[[c for c in data.columns if c.startswith('x')]].to_numpy().T
        u = data[[c for c in data.columns if c.startswith('u')]].to_numpy().T
        if np.any([c.startswith('p') for c in data.columns]):
            p = data[[c for c in data.columns if c.startswith('p')]]
            p = p.to_numpy().T
        else:
            p = None
        if 'v' in data.columns:
            v = data['v'].to_numpy()
        else:
            v = None
    elif isinstance(data, dict):
        t = data['t']
        x = _stack_columns(*[data[c] for c in data.keys() if c.startswith('x')])
        u = _stack_columns(*[data[c] for c in data.keys() if c.startswith('u')])
        if np.any([c.startswith('p') for c in data.keys()]):
            p = [data[c] for c in data.keys() if c.startswith('p')]
            p = _stack_columns(*p)
        else:
            p = None
        v = data.get('v', None)
    else:
        raise TypeError('data must be a DataFrame or dict')

    return t, x, u, p, v


def _stack_columns(*x):
    if len(x) == 1:
        return x[0]
    elif np.ndim(x[0]) == 1:
        return np.stack(x, axis=1)
    else:
        return np.stack(x, axis=-1)


def stack_dataframes(*data_list):
    """
    Extract `numpy` arrays from a list of dicts or `DataFrame`s (formatted by
    `pack_dataframe`), and concatenate these into a single set of arrays.

    Parameters
    ----------
    *data_list : list of dicts or DataFrames
        Each element of `data_list` is a dict or `DataFrame` with keys/columns

            * 't' : Time values of each data point (row).
            * 'x1', ..., 'xn' or 'x' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u1', ..., 'um' or 'u' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p1', ..., 'pn' or 'p' : Costates $p_1(t)$, ..., $p_n(t)$.
            * 'v' : Value function/cost-to-go $v(t)$.

    Returns
    -------
    t : (n_data,) array
        The concatenation of all 't' columns in `data_list`.
    x : (n_states, n_data) array
        The concatenation of all 'x' or 'xi' columns in `data_list`.
    u : (n_controls, n_data) array
        The concatenation of all 'u' or 'ui' columns in `data_list`.
    p : (n_states, n_data) array
        The concatenation of all 'p' or 'pi' columns in `data_list`.
    v : (n_points,) array
        The concatenation of all 'v' columns in `data_list`.
    """
    t = []
    x = []
    u = []
    p = []
    v = []

    for data in data_list:
        _t, _x, _u, _p, _v = unpack_dataframe(data)
        t.append(_t)
        x.append(_x)
        u.append(_u)
        if _p is not None:
            p.append(_p)
        if _v is not None:
            v.append(_v)

    t = np.concatenate(t)
    x = np.hstack(x)
    u = np.hstack(u)
    if len(p) >= 1:
        p = np.hstack(p)
    else:
        p = None
    if len(v) >= 1:
        v = np.concatenate(v)
    else:
        v = None

    return t, x, u, p, v


def save_data(data, filepath, overwrite=True):
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
            * 'x' or 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u' or 'u1', ..., 'um' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p' or 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t)$,
                optional.
            * 'v' : Value function/cost-to-go $v(t)$, optional.
    filepath : path-like
        Where the csv file should be saved.
    overwrite : bool, default=True
        If True, overwrite the csv file at `filepath`, if it exists. If False,
        append the data to the end of the existing csv.
    """
    t, x, u, p, v = stack_dataframes(*data)
    data = pack_dataframe(t, x, u, p, v)

    if not overwrite:
        try:
            existing_data = pd.read_csv(filepath)
            data = pd.concat([existing_data, data])
        except FileNotFoundError:
            pass

    data.to_csv(filepath, index=False)


def load_data(filepath, unpack=True):
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
        Each element of `data` is a dict or `DataFrame` with keys/columns

            * 't' : Time values of each data point (row).
            * 'x' or 'x1', ..., 'xn' : States $x_1(t)$, ..., $x_n(t)$.
            * 'u' or 'u1', ..., 'um' : Controls $u_1(t)$, ..., $u_m(t)$.
            * 'p' or 'p1', ..., 'pn' : Costates $p_1(t)$, ..., $p_n(t)$.
            * 'v' : Value function/cost-to-go $v(t)$.
    unpack : bool, default=True
        If True (default), returns a list of dict which includes 2d arrays for
        'x', 'u', and 'p'. If False, returns a list of `DataFrame`s with
        individual columns for each dimension of 'x', 'u', 'p'.
    """
    dataframe = pd.read_csv(filepath)
    # Find where trajectories start
    t0_idx = np.where(dataframe['t'].to_numpy() == 0.)[0]
    # Assume trajectories end before the start of the next trajectory
    # Pandas includes the ends of index slices, so subract 1 from these
    t1_idx = np.concatenate((t0_idx[1:] - 1, [len(dataframe)]))

    data = []
    for i0, i1 in zip(t0_idx, t1_idx):
        traj_data = dataframe.loc[i0:i1].reset_index(drop=True)
        if unpack:
            traj_data = dict(zip(['t', 'x', 'u', 'p', 'v'],
                                 unpack_dataframe(traj_data)))
        data.append(traj_data)

    return np.array(data, dtype=object)
