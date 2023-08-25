import numpy as np
from scipy import optimize, sparse
from scipy.interpolate import interp1d

_order_err_msg = ("order must be one of 'C' (C, row-major) or 'F' "
                  "(Fortran, column-major)")


def interp_guess(t, x, u, tau, time_map):
    '''
    Interpolate initial guesses for the state X and control U in physical time
    to Radau points in [-1,1).

    Parameters
    ----------
    t : (n_points,) array
        Time points for initial guess. Must be a strictly increasing sequence of
        real numbers with t[0]=0 and t[-1]=t1 > 0.
    X : (n_states, n_points) array
        Initial guess for the state values X(t).
    U : (n_controls, n_points) array
        Initial guess for the control values U(X(t)).
    tau : (n_nodes,) array
        Radau points computed by legendre_gauss_radau.make_LGR_nodes.

    Returns
    -------
    X : (n_states, n_nodes) array
        Interpolated state values X(tau).
    U : (n_controls, n_points) array
        Interpolated control values U(tau).
    '''
    t_mapped = time_map(np.reshape(t, (-1,)))
    x, u = np.atleast_2d(x), np.atleast_2d(u)

    x = interp1d(t_mapped, x, bounds_error=False, fill_value=x[:, -1])
    u = interp1d(t_mapped, u, bounds_error=False, fill_value=u[:, -1])

    return x(tau), u(tau)


def make_reshaping_funs(n_states, n_controls, n_nodes, order='C'):
    '''
    Generates a function which collects the state and control arrays into a
    single vector and a function which reverts the process.

    Parameters
    ----------
    n_states : int
        Number of state variables.
    n_controls : int
        Number of control variables.
    n_nodes : int
        Number of LGR collocation points.
    order : {'C', 'F'}, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering.

    Returns
    -------
    collect_vars : callable
        Function which returns XU = collect_vars(X, U), where X is an
        (n_states, n_nodes) array, U is an (n_controls, n_nodes) array, and XU
        is an ((n_states+n_controls)*n_nodes,) array.
    separate_vars : callable
        Function which returns X, U = separate_vars(XU), where XU
        is an ((n_states+n_controls)*n_nodes,) array, X is an
        (n_states, n_nodes) array, and U is an (n_controls, n_nodes) array.
    '''
    if order not in ['C', 'F']:
        raise ValueError(_order_err_msg)

    nx = n_states * n_nodes

    def collect_vars(X, U):
        return np.concatenate((
            X.flatten(order=order), U.flatten(order=order)
        ))

    def separate_vars(XU):
        X = XU[:nx].reshape((n_states, n_nodes), order=order)
        U = XU[nx:].reshape((n_controls, n_nodes), order=order)
        return X, U

    return collect_vars, separate_vars


def make_dynamic_constraint(dynamics, D, n_states, n_controls, separate_vars,
                            jac='2-point', order='C'):
    '''
    Create a function to evaluate the dynamic constraint DX - F(X,U) = 0 and its
    Jacobian. The Jacobian is returned as a callable which employs sparse
    matrices. In particular, the constraints at time tau_i are independent from
    constraints at time tau_j and some of the constraint Jacobians are constant.

    Parameters
    ----------
    dynamics : callable
        Right-hand side of the system, dXdt = dynamics(X,U).
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix.
    n_states : int
        Number of state variables.
    n_controls : int
        Number of control variables.
    separate_vars : callable
        Function which returns X, U = separate_vars(XU), where XU
        is an ((n_states+n_controls)*n_nodes,) array, X is an
        (n_states, n_nodes) array, and U is an (n_controls, n_nodes) array.
    jac : {callable, '3-point', '2-point', 'cs'}, default='2-point'
        Jacobian of the dynamics dXdt=F(X,U) with respect to states X and
        controls U. If callable, function dynamics_jac should take two arguments
        X and U with respective shapes (n_states, n_nodes) and
        (n_controls, n_nodes), and return a tuple of Jacobian arrays
        (dF/dX, dF/dU) with respective shapes (n_states, n_states, n_nodes) and
        (n_states, n_controls, n_nodes). Other string options specify the finite
        difference methods to use if the analytical Jacobian is not available.
    order : {'C', 'F'}, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering.

    Returns
    -------
    constraints : NonlinearConstraint
        Instance of scipy.optimize.NonlinearConstraint containing the constraint
        function and its sparse Jacobian.
    '''
    n_nodes = D.shape[0]
    DT = D.T

    # Dynamic constraint function evaluates to 0 when (X, U) is feasible
    def constr_fun(XU):
        X, U = separate_vars(XU)
        F = dynamics(X, U)
        DX = np.matmul(X, DT)
        return (DX - F).flatten(order=order)

    # Generates linear component of Jacobian
    if order == 'C':
        linear_constr = sparse.kron(sparse.identity(n_states), D)
    elif order == 'F':
        linear_constr = sparse.kron(D, sparse.identity(n_states))
    else:
        raise ValueError(_order_err_msg)

    linear_constr = sparse.hstack((
        linear_constr, np.zeros((n_states*n_nodes, n_controls*n_nodes))
    ))

    # Make constraint Jacobian function by summing linear and nonlinear parts
    if callable(jac):
        def constr_jac(XU):
            X, U = separate_vars(XU)
            dFdX, dFdU = jac(X, U)

            if order == 'C':
                dFdX = sparse.vstack([
                    sparse.diags(
                        diagonals=-dFdX[i],
                        offsets=range(0, n_nodes*n_states, n_nodes),
                        shape=(n_nodes, n_nodes*n_states)
                    ) for i in range(n_states)
                ])
                dFdU = sparse.vstack([
                    sparse.diags(
                        diagonals=-dFdU[i],
                        offsets=range(0, n_nodes*n_controls, n_nodes),
                        shape=(n_nodes, n_nodes*n_controls)
                    ) for i in range(n_states)
                ])
            elif order == 'F':
                dFdX = np.transpose(dFdX, (2,0,1))
                dFdU = np.transpose(dFdU, (2,0,1))
                dFdX = sparse.block_diag(-dFdX)
                dFdU = sparse.block_diag(-dFdU)

            Jac = sparse.hstack((dFdX, dFdU))

            return Jac + linear_constr
    else:
        # Generate sparsity structure for finite differences
        if order == 'C':
            sparsity = sparse.identity(n_nodes)
            sparsity = sparse.hstack([sparsity]*(n_states + n_controls))
            sparsity = sparse.vstack([sparsity]*n_states)
        elif order == 'F':
            sparsity = sparse.hstack((
                sparse.block_diag([np.ones((n_states, n_states))]*n_nodes),
                sparse.block_diag([np.ones((n_states, n_controls))]*n_nodes)
            ))

        def dynamics_wrapper(XU):
            X, U = separate_vars(XU)
            F = dynamics(X, U)
            return -F.flatten(order=order)

        def constr_jac(XU):
            # Compute nonlinear components with finite differences
            Jac = optimize._numdiff.approx_derivative(
                dynamics_wrapper, XU, sparsity=sparsity, method=jac
            )
            return Jac + linear_constr

    return optimize.NonlinearConstraint(
        fun=constr_fun, jac=constr_jac, lb=0., ub=0.
    )

def make_initial_condition_constraint(X0, n_controls, n_nodes, order='C'):
    '''
    Create a function which evaluates the initial condition constraint
    X(0) - X0 = 0. This is a linear constraint which is expressed by matrix
    multiplication.

    Parameters
    ----------
    X0 : (n_states,1) array
        Initial condition.
    n_controls : int
        Number of control variables.
    n_nodes : int
        Number of LGR collocation points.
    order : str, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering.

    Returns
    -------
    constraints : LinearConstraint
        Instance of scipy.optimize.LinearConstraint containing the constraint
        matrix and initial condition.
    '''
    X0_flat = np.reshape(X0, (-1,))
    n_states = X0_flat.shape[0]

    # Generates sparse matrix for multiplying combined state
    if order == 'C':
        A = sparse.eye(m=1, n=n_nodes)
        A = sparse.kron(sparse.identity(n_states), A)
        A = sparse.hstack((A, np.zeros((n_states, n_controls*n_nodes))))
    elif order == 'F':
        A = sparse.eye(m=n_states, n=(n_states+n_controls)*n_nodes)
    else:
        raise ValueError(_order_err_msg)

    return optimize.LinearConstraint(A=A, lb=X0_flat, ub=X0_flat)


def make_bound_constraint(u_lb, u_ub, n_states, n_nodes, order='C'):
    '''
    Create the control saturation constraints for all controls. Returns None if
    both control bounds are None.

    Parameters
    ----------
    u_lb : (n_controls,1) array or None
        Lower bounds for the controls.
    u_ub : (n_controls,1) array or None
        Upper bounds for the controls.
    n_states : int
        Number of state variables.
    n_nodes : int
        Number of LGR collocation points.
    order : str, default='C'
        Use C ('C', row-major) or Fortran ('F', column-major) ordering.

    Returns
    -------
    None
        Returned if both u_lb and u_ub are None.
    constraints : Bounds
        Instance of scipy.optimize.LinearBounds containing the control bounds
        mapped to the decision vector of states and controls. Returned only if
        at least one of u_lb and u_ub is not None.
    '''
    if u_lb is None and u_ub is None:
        return

    if u_lb is None:
        u_lb = np.full_like(u_ub, -np.inf)
    elif u_ub is None:
        u_ub = np.full_like(u_lb, np.inf)

    u_lb = np.reshape(u_lb, (-1,1))
    u_ub = np.reshape(u_ub, (-1,1))

    lb = np.concatenate((
        np.full(n_states*n_nodes, -np.inf),
        np.tile(u_lb, (1,n_nodes)).flatten(order=order)
    ))
    ub = np.concatenate((
        np.full(n_states*n_nodes, np.inf),
        np.tile(u_ub, (1,n_nodes)).flatten(order=order)
    ))

    return optimize.Bounds(lb=lb, ub=ub)
