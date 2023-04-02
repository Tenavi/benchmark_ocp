class OpenLoopSolution:
    """
    Object containing the solution to an open loop optimal control problem for
    one initial condition. The solution is stored at the time points chosen by
    the solver, as well as an interpolant appropriate for the solution type.

    Attributes
    ----------
    t : `(n_points,)` array
        Time points returned by the open loop solver.
    x : `(n_states, n_points)` array
        Values of the optimal state trajectory at times `t`.
    u : `(n_controls, n_points)` array
        Values of the optimal control at times `t`.
    p : `(n_states, n_points)` array
        Values of the costate at times `t`.
    v : `(n_points,)` array
        The value function evaluated at the points `x`.
    status : int
        Reason for solver termination. `status==0` indicates success, other
        values indicated various failure modes. See `message` for details.
    message : string
        Human-readable description of `status`.
    """
    def __init__(self, t, x, u, p, v, status, message, **kwargs):
        self.t = t
        self.x = x
        self.u = u
        self.p = p
        self.v = v
        self.status = status
        self.message = message

    def __call__(self, t):
        """
        Interpolate the optimal solution at new times `t`.

        Parameters
        ----------
        t : `(n_points,)` array
            Time points at which to evaluate the continuous solution.

        Returns
        -------
        x : `(n_states, n_points)` array
            Values of the optimal state trajectory at times `t`.
        u : `(n_controls, n_points)` array
            Values of the optimal control at times `t`.
        p : `(n_states, n_points)` array
            Values of the costate at times `t`.
        v : `(n_points,)` array
            The value function evaluated at the points `x`.
        """
        raise NotImplementedError

    def check_convergence(self, fun, tol, verbose=False):
        """
        Compare the running cost of the solution at final time to a specified
        tolerance.

        Parameters
        ----------
        fun : callable
            Running cost function. See `OptimalControlProblem.running_cost`.
        tol : float
            The maximum allowable final time running cost. This can be negative
            for cost functions which are allowed to be negative (rare).
        verbose : bool, default=False
            Set `verbose=True` to print out the results.

        Returns
        -------
        converged : bool
            `True` if `self.status == 0` and
            `fun(self.x[:, -1], self.u[:, -1]) <= tol`.
        """
        converged = self.status == 0
        if not converged:
            if verbose:
                print(f'Solution failed to converge: status = {self.status:d}: '
                      f'{self.message}')
            return converged

        L = float(fun(self.x[:, -1], self.u[:, -1]))
        converged = converged and L <= tol
        if verbose:
            if converged:
                print(f'Solution converged: running cost = {L:1.2e} <= '
                      f'tolerance {tol:1.2e}')
            else:
                print(f'Solution failed to converge: running cost = {L:1.2e} > '
                      f'tolerance {tol:1.2e}')

        return converged
