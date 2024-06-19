import warnings

import numpy as np
from scipy.optimize._constraints import old_bound_to_new, _arr_to_scalar
from scipy.optimize._differentiable_functions import FD_METHODS
from scipy.optimize._minimize import (MemoizeJac, standardize_constraints,
                                      standardize_bounds, _add_to_array,
                                      _remove_from_bounds, _remove_from_func)
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._optimize import (OptimizeResult, _prepare_scalar_function,
                                      _check_clip_x)
from scipy.optimize._slsqp import slsqp


_epsilon = np.sqrt(np.finfo(float).eps)


def minimize(fun, x0, args=(), jac=None, bounds=None, constraints=(), tol=None,
             options=None):
    """
    Minimization of scalar function of one or more variables.

    Wrapper and modification of `scipy.optimize.optimize` implementing shortcuts
    to the 'SLSQP' method and extracting the KKT multipliers.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            fun(x, *args) -> float
        where x is an 1-D array with shape (n,) and args
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where n is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (fun, jac and hess functions).
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:
            jac(x, *args) -> array_like, shape (n,)
        where x is an array with shape (n,) and args is a tuple with
        the fixed parameters. If jac is a Boolean and is True, fun is
        assumed to return a tuple (f, g) containing the objective
        function and the gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified bounds.
    bounds : `scipy.optimize.Bounds`, optional
        Bounds on variables as an instance of `Bounds` class.
    constraints : (list of) `scipy.optimize.Constraint`, optional
        Constraints defined as a single object or a list of objects specifying
        constraints to the optimization problem.
        Available constraints are:
            - `LinearConstraint`
            - `NonlinearConstraint`
    tol : float, optional
        Tolerance for termination. When tol is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to tol. For detailed control, use solver-specific options.
    options : dict, optional
        A dictionary of solver options.
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
            ftol : float
                Precision goal for the value of f in the stopping criterion.
            eps: float
                Step size used for numerical approximation of the Jacobian.

    Returns
    -------
    res : `scipy.optimize.OptimizeResult`
        The optimization result represented as an `OptimizeResult` object.
        Important attributes are: `x`, the solution array; `success`, a
        Boolean flag indicating if the optimizer exited successfully; and
        `message`, which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.
    """
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.dtype.kind in np.typecodes['AllInteger']:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    # check gradient vector
    if callable(jac) or jac in FD_METHODS:
        pass
    elif jac is True:
        # fun returns func and grad
        fun = MemoizeJac(fun)
        jac = fun.derivative
    else:
        # default if jac option is not understood
        jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        options.setdefault('ftol', tol)

    constraints = standardize_constraints(constraints, x0, 'slsqp')

    remove_vars = False
    if bounds is not None:
        # SLSQP can't take the finite-difference derivatives when a variable is
        # fixed by the bounds. To avoid this issue, remove fixed variables from
        # the problem.

        # convert to new-style bounds so we only have to consider one case
        bounds = standardize_bounds(bounds, x0, 'new')

        # determine whether any variables are fixed
        i_fixed = bounds.lb == bounds.ub

        # determine whether finite differences are needed for any grad/jac
        fd_needed = not callable(jac)
        for con in constraints:
            if not callable(con.get('jac', None)):
                fd_needed = True

        # If finite differences are ever used, remove all fixed variables
        remove_vars = i_fixed.any() and fd_needed
        if remove_vars:
            x_fixed = bounds.lb[i_fixed]
            x0 = x0[~i_fixed]
            bounds = _remove_from_bounds(bounds, i_fixed)
            fun = _remove_from_func(fun, i_fixed, x_fixed)
            if callable(jac):
                jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)

            # make a copy of the constraints so the user's version doesn't
            # get changed. (Shallow copy is ok)
            constraints = [con.copy() for con in constraints]
            for con in constraints:  # yes, guaranteed to be a list
                con['fun'] = _remove_from_func(con['fun'], i_fixed, x_fixed,
                                               min_dim=1, remove=0)
                if callable(con.get('jac', None)):
                    con['jac'] = _remove_from_func(con['jac'], i_fixed, x_fixed,
                                                   min_dim=2, remove=1)
    bounds = standardize_bounds(bounds, x0, 'slsqp')

    res = _minimize_slsqp(fun, x0, args, jac, bounds, constraints, **options)

    if remove_vars:
        res.x = _add_to_array(res.x, i_fixed, x_fixed)
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        if 'hess_inv' in res:
            res.hess_inv = None

    return res


def _minimize_slsqp(fun, x0, args=(), jac=None, bounds=None, constraints=(),
                    maxiter=100, ftol=1e-06, iprint=1, disp=False, eps=_epsilon,
                    finite_diff_rel_step=None):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP). Modified from
    `scipy.optimize._slsqp_py._minimize_slsqp` to extract KKT multipliers.
    Based on work by github user andyfaff.

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    iter = maxiter - 1
    acc = ftol

    if not disp:
        iprint = 0

    x = np.asarray(x0).flatten()

    # SLSQP is sent 'old-style' bounds, 'new-style' bounds are required by
    # ScalarFunction
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # clip the initial guess to bounds, otherwise ScalarFunction doesn't work
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', "outside bounds",
                                                RuntimeWarning)
                        x = _check_clip_x(x, new_bounds)

                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args,
                                                 rel_step=finite_diff_rel_step,
                                                 bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point',
                                                 abs_step=eps, args=args,
                                                 bounds=new_bounds)

                return cjac
            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Set the parameters that SLSQP will need
    # _meq_cv: a list containing the length of values each constraint function
    _meq_cv = [len(np.atleast_1d(c['fun'](x, *c['args']))) for c in cons['eq']]
    _mieq_cv = [len(np.atleast_1d(c['fun'](x, *c['args']))) for c in cons['ineq']]
    # meq, mieq: number of equality and inequality constraints
    meq = sum(_meq_cv)
    mieq = sum(_mieq_cv)
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = np.array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = np.zeros(len_w)
    jw = np.zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = np.array(
            [(_arr_to_scalar(l), _arr_to_scalar(u)) for (l, u) in bounds],
            dtype=float
        )
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~np.isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # ScalarFunction provides function and gradient evaluation
    sf = _prepare_scalar_function(fun, x, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
    # doesn't get sent to the func/grad evaluator.
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # Initialize the iteration counter and the mode value
    mode = np.array(0, int)
    acc = np.array(acc, float)
    majiter = np.array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = np.array(0, float)
    f0 = np.array(0, float)
    gs = np.array(0, float)
    h1 = np.array(0, float)
    h2 = np.array(0, float)
    h3 = np.array(0, float)
    h4 = np.array(0, float)
    t = np.array(0, float)
    t0 = np.array(0, float)
    tol = np.array(0, float)
    iexact = np.array(0, int)
    incons = np.array(0, int)
    ireset = np.array(0, int)
    itermx = np.array(0, int)
    line = np.array(0, int)
    n1 = np.array(0, int)
    n2 = np.array(0, int)
    n3 = np.array(0, int)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from
    # ScalarFunction
    fx = wrapped_fun(x)
    g = np.append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # Call SLSQP
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # objective and constraint evaluation required
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)

        if mode == -1:  # gradient evaluation required
            g = np.append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        if majiter > majiter_prev:
            # Print the status of the current iterate if iprint > 2
            if iprint >= 2:
                print("%5i %5i % 16.6E % 16.6E" % (majiter, sf.nfev,
                                                   fx, np.linalg.norm(g)))

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)

    # Obtain KKT multipliers
    im = 1
    il = im + la
    ix = il + (n1*n)//2 + 1
    ir = ix + n - 1
    _kkt_mult = w[ir:ir + m]

    # KKT multipliers
    w_ind = 0
    kkt_multiplier = dict()

    for _t, cv in [("eq", _meq_cv), ("ineq", _mieq_cv)]:
        kkt = []

        for dim in cv:
            kkt += [_kkt_mult[w_ind:(w_ind + dim)]]
            w_ind += dim

        kkt_multiplier[_t] = kkt

    # Optimization loop complete. Print status if requested
    if iprint >= 1:
        print(f"{exit_modes[int(mode)]}    (Exit mode {mode})")
        print("            Current function value:", fx)
        print("            Iterations:", majiter)
        print("            Function evaluations:", sf.nfev)
        print("            Gradient evaluations:", sf.ngev)

    return OptimizeResult(x=x, fun=fx, jac=g[:-1],
                          nit=int(majiter),
                          nfev=sf.nfev, njev=sf.ngev, status=int(mode),
                          message=exit_modes[int(mode)],
                          success=(mode==0),
                          kkt=kkt_multiplier)


def _eval_constraint(x, cons):
    # Compute constraints
    if cons['eq']:
        c_eq = np.concatenate(
            [np.atleast_1d(con['fun'](x, *con['args'])) for con in cons['eq']]
        )
    else:
        c_eq = np.zeros(0)

    if cons['ineq']:
        c_ieq = np.concatenate(
            [np.atleast_1d(con['fun'](x, *con['args'])) for con in cons['ineq']]
        )
    else:
        c_ieq = np.zeros(0)

    # Now combine c_eq and c_ieq into a single matrix
    c = np.concatenate((c_eq, c_ieq))
    return c


def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    # Compute the normals of the constraints
    if cons['eq']:
        a_eq = np.vstack(
            [con['jac'](x, *con['args']) for con in cons['eq']]
        )
    else:  # no equality constraint
        a_eq = np.zeros((meq, n))

    if cons['ineq']:
        a_ieq = np.vstack(
            [con['jac'](x, *con['args']) for con in cons['ineq']]
        )
    else:  # no inequality constraint
        a_ieq = np.zeros((mieq, n))

    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = np.zeros((la, n))
    else:
        a = np.vstack((a_eq, a_ieq))
    a = np.concatenate((a, np.zeros([la, 1])), 1)

    return a


def _clip_x_for_func(func, bounds):
    # ensures that x values sent to func are clipped to bounds

    # this is used as a mitigation for gh11403, slsqp/tnc sometimes
    # suggest a move that is outside the limits by 1 or 2 ULP. This
    # unclean fix makes sure x is strictly within bounds.
    def eval(x):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "Values in x were outside bounds",
                                    RuntimeWarning)
            x = _check_clip_x(x, bounds)
        return func(x)

    return eval
