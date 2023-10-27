import numpy as np


class OpenLoopSolution:
    """
    Object containing the solution to an open loop optimal control problem for
    one initial condition. The solution is stored at the time points chosen by
    the solver, as well as an interpolant appropriate for the solution type.
    """
    def __init__(self, t, x, u, p, v, status, message, **kwargs):
        self.t = np.asarray(t)
        """(n_points,) array. Time points returned by the open loop solver."""
        self.x = np.asarray(x)
        """(n_states, n_points) array. The optimal state trajectory."""
        self.u = np.asarray(u)
        """(n_controls, n_points) array. The optimal control profile."""
        self.p = np.asarray(p)
        """(n_states, n_points) array. The costate trajectory."""
        self.v = np.asarray(v)
        """(n_points,) array. The value function evaluated at the points `x`."""
        self.status = int(status)
        """int. Reason for solver termination. `status==0` indicates success;
        other values indicate various failure modes. See `message` for details.
        """
        self.message = str(message)
        """str. Human-readable description of `status`."""

    def __call__(self, t):
        """
        Interpolate the optimal solution at new times `t`.

        Parameters
        ----------
        t : (n_points,) array
            Time points at which to evaluate the continuous solution.

        Returns
        -------
        x : (n_states, n_points) array
            Values of the optimal state trajectory at times `t`.
        u : (n_controls, n_points) array
            Values of the optimal control at times `t`.
        p : (n_states, n_points) array
            Values of the costate at times `t`.
        v : (n_points,) array
            The value function evaluated at the points `x`.
        """
        raise NotImplementedError

    def check_convergence(self, fun, tol, verbose=False):
        """
        Compare the running cost of the solution at final time to a specified
        tolerance. Relevant for finite horizon approximations of infinite
        horizon problems to determine if the solution has converged.

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
            `True` if `self.status==0` and
            `fun(self.x[:, -1], self.u[:, -1]) <= tol`.
        """
        converged = self.status == 0
        if not converged:
            if verbose:
                print(f"Solution failed to converge: status = {self.status:d}: "
                      f"{self.message}")
            return converged

        L = float(fun(self.x[:, -1], self.u[:, -1]))
        converged = converged and L <= tol
        if verbose:
            if converged:
                print(f"Solution converged: running cost = {L:1.2e} <= "
                      f"tolerance {tol:1.2e}")
            else:
                print(f"Solution failed to converge: running cost = {L:1.2e} > "
                      f"tolerance {tol:1.2e}")

        return converged


class AntialiasedSolution(OpenLoopSolution):
    def __init__(self, sols, t_break=[]):
        """

        Parameters
        ----------
        sols
        t_break
        """
        if not hasattr(sols, '__iter__'):
            sols = [sols]
        if len(sols) < 1:
            raise ValueError("sols must be a non-empty list")
        if any([not isinstance(sol, OpenLoopSolution) for sol in sols]):
            raise TypeError("sols must be a list of OpenLoopSolutions")

        self._sols = sols

        self._t_break = np.asarray(t_break, dtype=float).reshape(-1, 1)

        if self._t_break.size != self.n_segments - 1:
            raise ValueError(f"len(t_break) = {self._t_break.size} but must be "
                             f"len(sols) - 1 = {self.n_segments - 1}")
        if not np.all(self._t_break[:-1] < self._t_break[1:]):
            raise ValueError("t_break must be sorted in ascending order")

        self._t_break_extended = np.vstack((self._t_break, np.inf))

        combined_sol = {'t': [], 'x': [], 'u': [], 'p': [], 'v': [],
                        'status': self._sols[-1].status,
                        'message': self._sols[-1].message}

        t0 = 0.
        for k, sol in enumerate(self._sols):
            t1 = self._t_break_extended[k]
            # Individual solutions start at t=0, so shift time by the previous
            # breakpoint
            t = sol.t + t0

            # Find which parts of the solution are included
            idx = t < t1

            combined_sol['t'].append(t[idx])

            for key in ['x', 'u', 'p', 'v']:
                combined_sol[key].append(getattr(sol, key)[..., idx])

            t0 = t1

        # Make sure that the value function decreases over time by including
        # contributions from segments that follow the current segment.
        # We only need to do this for the first n - 1 segments.
        self._v_diff = np.empty(self.n_segments - 1)
        for k in range(self.n_segments - 2, -1, -1):
            # Find what this segment thinks the value should be at the end
            t1 = self._t_break_extended[k]
            _, _, _, v1 = self._sols[k](t1)
            # If needed, raise the value by the value difference of this segment
            # and the start of the following segment
            self._v_diff[k] = combined_sol['v'][k + 1][0] - np.squeeze(v1)
            if self._v_diff[k] > 0.:
                combined_sol['v'][k] = combined_sol['v'][k] + self._v_diff[k]

        for key in ['t', 'x', 'u', 'p', 'v']:
            combined_sol[key] = np.concatenate(combined_sol[key], axis=-1)

        super().__init__(**combined_sol)

    @property
    def n_segments(self):
        """int. The number of individual solutions joined together."""
        return len(self._sols)

    def _assign_to_segments(self, t):
        """
        Compute a set of logical indices binning time values `t` between
        time breakpoints `self._t_break`. Explicitly, these would be computed as
        ```
        idx[0] = t < self._t_break[0]
        for i in range(1, self.n_segments - 1):
            idx[i] = self._t_break[i - 1] <= t < self._t_break[i]
        idx[-1] = self._t_break[-1] <= t
        ```
        but the computation is vectorized.

        Parameters
        ----------
        t : (n_points,) array
            Time points to bin.

        Returns
        -------
        idx : (`n_segments`, n_points) bool array
            Boolean array such that `t[idx[i]]` is the subset of `t` which
            belong to the `i`th solution segment.
        """
        t = np.asarray(t).reshape(-1)
        idx = t < self._t_break_extended
        idx[1:] = np.logical_and(idx[1:], self._t_break <= t)
        return idx

    def __call__(self, t):
        t = np.asarray(t).reshape(-1)
        n_t = t.shape[0]

        indices = self._assign_to_segments(t)

        x = np.empty((self.x.shape[0], n_t))
        u = np.empty((self.u.shape[0], n_t))
        p = np.empty((self.p.shape[0], n_t))
        v = np.empty(n_t)

        # Loop over segments
        for k, idx in enumerate(indices):
            _t = t[idx]
            if k >= 1:
                # Subtract the segment start time, since individual
                # interpolators expect time to start at 0
                _t = _t - self._t_break[k - 1]
            if _t.size > 0:
                x[:, idx], u[:, idx], p[:, idx], v[idx] = self._sols[k](_t)
                if k < self.n_segments - 1 and self._v_diff[k] > 0.:
                    v[idx] += self._v_diff[k]

        return x, u, p, v
