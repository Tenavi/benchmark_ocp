import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, NonlinearConstraint
from scipy.interpolate import make_interp_spline

from optimalcontrol.utilities import resize_vector


_order_err_msg = ("order must be one of 'C' (C, row-major) or 'F' "
                  "(Fortran, column-major)")


def setup(ocp, x0, tau, w, D, order='F'):
    """
    Wrapper function which calls other functions in this module to build a
    collocated objective function, dynamic constraints, and control constraints.

    Parameters
    ----------
    ocp : OptimalControlProblem
        The optimal control problem to solve.
    x0 : (n_states,) array
        Initial condition.
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points `tau`.
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points
        `tau`.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    obj_fun : callable
        Function of the combined decision variables `xu` returning the
        integrated running cost and its Jacobian with respect to `xu`. See
        `make_objective_fun`.
    dynamic_constraint : `scipy.optimize.NonlinearConstraint`
        Instance of `NonlinearConstraint` containing the constraint function and
        its sparse Jacobian. See `make_dynamic_constraint`.
    bound_constraint : `scipy.optimize.Bounds` or None
        Instance of `Bounds` containing the control bounds mapped to the
        decision vector of states and controls. See `make_bound_constraint`.
    """
    # Quadrature integration of running cost
    cost_fun = make_objective_fun(ocp, w, order=order)

    # Dynamic constraint
    dyn_constr = make_dynamic_constraint(ocp, D, order=order)

    # Control and initial condition constraints
    bound_constr = make_bound_constraint(ocp, x0, tau.shape[0], order=order)

    return cost_fun, dyn_constr, bound_constr


def make_objective_fun(ocp, w, order='F'):
    """
    Create a function to evaluate the quadrature-integrated running cost,
    `dot(w, ocp.running_cost(x, u))`, and its Jacobian.

    Parameters
    ----------
    ocp : OptimalControlProblem
        The optimal control problem to solve.
    w : (n_nodes,) array
        LGR quadrature weights.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    obj_fun : callable
        Function of the combined decision variables `xu` returning the
        integrated running cost and its Jacobian with respect to `xu`.
    """
    def obj_fun(xu):
        x, u = separate_vars(xu, ocp.n_states, ocp.n_controls, order=order)

        L = ocp.running_cost(x, u)
        dLdx, dLdu = ocp.running_cost_grad(x, u, L0=L)

        cost = np.dot(L, w)
        jac = collect_vars(dLdx * w, dLdu * w, order=order)

        return cost, jac

    return obj_fun


def make_dynamic_constraint(ocp, D, order='F'):
    """
    Create a function to evaluate the dynamic constraint,
    `ocp.dynamics(x, u) - D @ x == 0`, and its Jacobian. The Jacobian is
    returned as a callable which employs sparse matrices. In particular, the
    constraints at time `tau[i]` are independent of constraints at time
    `tau[j]`, and some constraint Jacobians are constant.

    Parameters
    ----------
    ocp : OptimalControlProblem
        The optimal control problem to solve.
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    dynamic_constraint : `scipy.optimize.NonlinearConstraint`
        Instance of `NonlinearConstraint` containing the constraint function and
        its sparse Jacobian.
    """
    def dynamics_constr_fun(x, u):
        f = ocp.dynamics(x, u)
        Dx = np.matmul(x, D.T)
        return f - Dx

    return make_nonlinear_constraint(ocp.n_states, ocp.n_controls, D.shape[0],
                                     dynamics_constr_fun, jac=ocp.jac,
                                     linear_part=-D, order=order)


def make_nonlinear_constraint(n_states, n_controls, n_nodes, fun, jac=None,
                              linear_part=None, lb=0., ub=0., order='F'):
    """
    Utility function for applying general nonlinear constraint functions to
    state and control decision vectors

    Parameters
    ----------
    n_states : int
        Number of states.
    n_controls : int
        Number of controls.
    n_nodes : int
        Number of LGR collocation points.
    fun : callable
        Constraint function to wrap. Expects the call signature `c = fun(x, u)`,
        where `x` and `u` are separated states and controls with respective
        shapes `(n_states, n_nodes)` and `(n_controls, n_nodes)`, and `c` is an
        `(n_constraints, n_nodes)` array of constraints for which we desire
        `lb <= c <= ub`.
    jac : callable, optional
        Jacobians of `fun` with respect to `x` and `u`. Expects the call
        signature `dcdx, dcdu = jac(x, u)`, where `dcdx` is the
        `(n_constraints, n_states, n_nodes)` array of Jacobians with respect to
        each state `x` and `dcdu` is the `(n_constraints, n_states, n_nodes)`
        array of Jacobians with respect each control `u`. If not provided, uses
        numerical Jacobians.
    linear_part : (n_nodes, n_nodes) array, optional
        Linear part of the Jacobian with respect to states. If provided, assumes
        that `fun` is equal to some nonlinear part plus
        `matmul(x, linear_part.T)`. When evaluating the constraint Jacobian,
        we sum the state Jacobian `dcdx` returned by `jac` with `linear_part`.
        This separation can improve computational efficiency.
    lb : (n_constraints,) array, default=0
        Desired lower bound of the constraint function.
    ub : (n_constraints,) array, default=0
        Desired upper bound of the constraint function.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    constraint : `scipy.optimize.NonlinearConstraint`
        Instance of `NonlinearConstraint` containing the constraint function and
        its sparse Jacobian.
    """
    n_x, n_u, n_t = n_states, n_controls, n_nodes

    def constr_fun(xu):
        x, u = separate_vars(xu, n_x, n_u, order=order)
        return fun(x, u).reshape(-1, order=order)

    # Generates linear component of Jacobian
    if linear_part is not None:
        if order == 'F':
            linear_part = sparse.kron(linear_part, sparse.identity(n_x))
        elif order == 'C':
            linear_part = sparse.kron(sparse.identity(n_x), linear_part)
        else:
            raise ValueError(_order_err_msg)

    # Make constraint Jacobian function by summing linear and nonlinear parts
    if jac is None:
        constr_jac = None
    else:
        def constr_jac(xu):
            x, u = separate_vars(xu, n_x, n_u, order=order)
            dfdx, dfdu = jac(x, u)

            if order == 'F':
                dfdx = np.transpose(dfdx, (2, 0, 1))
                dfdu = np.transpose(dfdu, (2, 0, 1))
                dfdx = sparse.block_diag(dfdx)
                dfdu = sparse.block_diag(dfdu)
            elif order == 'C':
                dfdx = sparse.vstack(
                    [sparse.diags(diagonals=dfdx[i],
                                  offsets=range(0, n_t * n_x, n_t),
                                  shape=(n_t, n_t * n_x))
                     for i in range(n_x)])
                dfdu = sparse.vstack(
                    [sparse.diags(diagonals=dfdu[i],
                                  offsets=range(0, n_t * n_u, n_t),
                                  shape=(n_t, n_t * n_u))
                     for i in range(n_x)])

            if linear_part is not None:
                dfdx += linear_part

            return sparse.hstack((dfdx, dfdu))

    return NonlinearConstraint(fun=constr_fun, jac=constr_jac, lb=lb, ub=ub)


def make_bound_constraint(ocp, x0, n_nodes, order='F'):
    """
    Create a `Bounds` object containing control saturation constraints, initial
    condition constraints, and simple bounds on states.

    Parameters
    ----------
    ocp : OptimalControlProblem
        The optimal control problem to solve.
    x0 : (n_states,) array
        Initial condition.
    n_nodes : int
        Number of LGR collocation points.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    bound_constraint : `scipy.optimize.Bounds`
        Instance of `Bounds` containing state and control bounds and the initial
        condition constraint mapped to the state and control decision vector.
    """
    x0 = np.reshape(x0, (ocp.n_states,))

    n_x, n_u = ocp.n_states, ocp.n_controls

    x_lb = _standardize_bound_vector(ocp.state_lb, -np.inf, n_x, n_nodes)
    x_ub = _standardize_bound_vector(ocp.state_ub, np.inf, n_x, n_nodes)

    x_lb[:, 0] = x0
    x_ub[:, 0] = x0

    u_lb = _standardize_bound_vector(ocp.control_lb, -np.inf, n_u, n_nodes)
    u_ub = _standardize_bound_vector(ocp.control_ub, np.inf, n_u, n_nodes)

    return Bounds(lb=collect_vars(x_lb, u_lb, order=order),
                  ub=collect_vars(x_ub, u_ub, order=order))


def _standardize_bound_vector(bound, fill_value, n_expect, n_nodes):
    if bound is None:
        bound = np.full((n_expect,), fill_value)
    else:
        bound = resize_vector(bound, n_expect)

    return np.tile(np.reshape(bound, (n_expect, 1)), (1, n_nodes))


def collect_vars(x, u, order='F'):
    """
    Gather separate state and control matrices arranged by (dimension, time)
    into a single 1d array for optimization, with states first and controls
    second.

    Parameters
    ----------
    x : (n_states, n_nodes) array
        States arranged by (dimension, time).
    u : (n_controls, n_nodes) array
        Controls arranged by (dimension, time).
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    xu : 1d array
        Array containing states `x` and controls `u`, with
        `xu[:x.size] == x.flatten(order=order)` and
        `xu[x.size:] == u.flatten(order=order)`.
    """
    return np.concatenate((x.reshape(-1, order=order),
                           u.reshape(-1, order=order)))


def separate_vars(xu, n_states, n_controls, order='F'):
    """
    Given a single 1d array containing states and controls at all time nodes,
    assembled using `collect_vars`, separate the array into states and controls
    and reshape these into 2d arrays arranged by (dimension, time).

    Parameters
    ----------
    xu : 1d array
        Array containing states `x` and controls `u`. `xu.size` must be
        divisible by `n_states + n_controls`, i.e.
        `xu.size == (n_states + n_controls) * n_nodes` for some int `n_nodes`.
    n_states : int
        Number of states.
    n_controls : int
        Number of controls.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    x : (n_states, n_nodes) array
        States extracted from `xu` arranged by (dimension, time).
    u : (n_controls, n_nodes) array
        Controls extracted from `xu` arranged by (dimension, time).
    """
    n_nodes = xu.size // (n_states + n_controls)
    nx = n_states * n_nodes
    x = xu[:nx].reshape((n_states, n_nodes), order=order)
    u = xu[nx:].reshape((n_controls, n_nodes), order=order)
    return x, u


def interp_guess(t, x, u, tau, inverse_time_map):
    """
    Linearly interpolate initial guesses for the state and control in physical
    time to collocation points. Points which need to be extrapolated beyond
    `inverse_time_map(tau) > t[-1]` simply use the final values for `x[:, -1]`
    and `u[:, -1]`.

    Parameters
    ----------
    t : (n_points,) array
        Time points for initial guess.
    x : (n_states, n_points) array
        Initial guess for the state values x(t).
    u : (n_controls, n_points) array
        Initial guess for the control values u(t).
    tau : (n_nodes,) array
        Radau points computed by `radau.make_lgr_nodes`.
    inverse_time_map : callable
        Function to map collocation points to physical time, e.g.
        `radau.time_map`.

    Returns
    -------
    x_interp : (n_states, n_nodes) array
        Interpolated states, `x(inverse_time_map(tau))`.
    u_interp : (n_controls, n_points) array
        Interpolated controls, `u(inverse_time_map(tau))`.
    """
    t = np.reshape(t, (-1,))
    x, u = np.atleast_2d(x), np.atleast_2d(u)

    x_interp = make_interp_spline(t, x, k=1, axis=-1)
    u_interp = make_interp_spline(t, u, k=1, axis=-1)

    tau_mapped = inverse_time_map(np.reshape(tau, (-1,)))
    extra_idx = tau_mapped > t[-1]

    x_interp = x_interp(tau_mapped)
    u_interp = u_interp(tau_mapped)

    x_interp[:, extra_idx] = x[:, -1:]
    u_interp[:, extra_idx] = u[:, -1:]

    return x_interp, u_interp
