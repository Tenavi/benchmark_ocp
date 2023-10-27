import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.interpolate import make_interp_spline

from optimalcontrol.utilities import resize_vector


_order_err_msg = ("order must be one of 'C' (C, row-major) or 'F' "
                  "(Fortran, column-major)")


def setup(ocp, tau, w, D, order='F'):
    """
    Wrapper function which calls other functions in this module to build a
    collocated objective function, dynamic constraints, and control constraints.

    Parameters
    ----------
    ocp : OptimalControlProblem
        The optimal control problem to solve.
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
        decision vector of states and controls. Returned only if at least one of
        `ocp.parameters.u_lb` and `ocp.parameters.u_ub` is not None. See
        `make_bound_constraint`.
    """
    # Quadrature integration of running cost
    cost_fun = make_objective_fun(ocp, w, order=order)

    # Dynamic and initial condition constraints
    dyn_constr = make_dynamic_constraint(ocp, D, order=order)

    # Control bounds
    u_lb = getattr(ocp.parameters, 'u_lb', None)
    u_ub = getattr(ocp.parameters, 'u_ub', None)
    if u_lb is not None:
        u_lb = resize_vector(u_lb, ocp.n_controls)
    if u_ub is not None:
        u_ub = resize_vector(u_ub, ocp.n_controls)

    bound_constr = make_bound_constraint(u_lb, u_ub, ocp.n_states, tau.shape[0],
                                         order=order)

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
    `ocp.dynamics(x, u) - D @ x== 0`, and its Jacobian. The Jacobian is
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
    n_t = D.shape[0]
    n_x, n_u = ocp.n_states, ocp.n_controls

    # Dynamic constraint function evaluates to 0 when (x, u) is feasible
    def constr_fun(xu):
        x, u = separate_vars(xu, n_x, n_u, order=order)
        f = ocp.dynamics(x, u)
        Dx = np.matmul(x, D.T)
        return (f - Dx).flatten(order=order)

    # Generates linear component of Jacobian
    if order == 'F':
        linear_part = sparse.kron(-D, sparse.identity(n_x))
    elif order == 'C':
        linear_part = sparse.kron(sparse.identity(n_x), -D)
    else:
        raise ValueError(_order_err_msg)

    # Make constraint Jacobian function by summing linear and nonlinear parts
    def constr_jac(xu):
        x, u = separate_vars(xu, n_x, n_u, order=order)
        dfdx, dfdu = ocp.jac(x, u)
        if order == 'F':
            dfdx = np.transpose(dfdx, (2, 0, 1))
            dfdu = np.transpose(dfdu, (2, 0, 1))
            dfdx = sparse.block_diag(dfdx)
            dfdu = sparse.block_diag(dfdu)
        elif order == 'C':
            dfdx = sparse.vstack([sparse.diags(diagonals=dfdx[i],
                                               offsets=range(0, n_t * n_x, n_t),
                                               shape=(n_t, n_t * n_x))
                                  for i in range(n_x)])
            dfdu = sparse.vstack([sparse.diags(diagonals=dfdu[i],
                                               offsets=range(0, n_t * n_u, n_t),
                                               shape=(n_t, n_t * n_u))
                                  for i in range(n_x)])

        return sparse.hstack((dfdx + linear_part, dfdu))

    return NonlinearConstraint(fun=constr_fun, jac=constr_jac, lb=0., ub=0.)


def make_initial_condition_constraint(x0, n_controls, n_nodes, order='F'):
    """
    Create a function which evaluates the initial condition constraint
    `x(0) - x0 == 0`. This is a linear constraint which is expressed by matrix
    multiplication.

    Parameters
    ----------
    x0 : (n_states,) array
        Initial condition.
    n_controls : int
        Number of control variables.
    n_nodes : int
        Number of LGR collocation points.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    constraints : `scipy.optimize.LinearConstraint`
        Instance of `LinearConstraint` containing the constraint matrix and
        initial condition.
    """
    x0 = np.reshape(x0, (-1,))
    n_states = x0.shape[0]

    # Generates sparse matrix for multiplying combined state
    if order == 'F':
        A = sparse.eye(m=n_states, n=(n_states + n_controls) * n_nodes)
    elif order == 'C':
        A = sparse.eye(m=1, n=n_nodes)
        A = sparse.kron(sparse.identity(n_states), A)
        A = sparse.hstack((A, np.zeros((n_states, n_controls * n_nodes))))
    else:
        raise ValueError(_order_err_msg)

    return LinearConstraint(A=A, lb=x0, ub=x0)


def make_bound_constraint(u_lb, u_ub, n_states, n_nodes, order='F'):
    """
    Create the control saturation constraints for all controls. Returns None if
    both control bounds are None.

    Parameters
    ----------
    u_lb : (n_controls, 1) array or None
        Lower bounds for the controls.
    u_ub : (n_controls, 1) array or None
        Upper bounds for the controls.
    n_states : int
        Number of state variables.
    n_nodes : int
        Number of LGR collocation points.
    order : {'C', 'F'}, default='F'
        Use C (row-major) or Fortran (column-major) ordering.

    Returns
    -------
    bound_constraint : `scipy.optimize.Bounds` or None
        Instance of `Bounds` containing the control bounds mapped to the
        decision vector of states and controls. Returned only if at least one of
        `u_lb` and `u_ub` is not None.
    """
    if u_lb is None and u_ub is None:
        return

    if u_lb is None:
        u_lb = np.full_like(u_ub, -np.inf)
    elif u_ub is None:
        u_ub = np.full_like(u_lb, np.inf)

    u_lb = np.reshape(u_lb, (-1, 1))
    u_ub = np.reshape(u_ub, (-1, 1))

    lb = np.concatenate((np.full(n_states * n_nodes, -np.inf),
                         np.tile(u_lb, (1, n_nodes)).flatten(order=order)))

    ub = np.concatenate((np.full(n_states * n_nodes, np.inf),
                         np.tile(u_ub, (1, n_nodes)).flatten(order=order)))

    return Bounds(lb=lb, ub=ub)


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
    return np.concatenate((x.flatten(order=order), u.flatten(order=order)))


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