class OpenLoopSolution:
    """
    Bunch object containing the solution to an open loop optimal control problem
    for one initial condition.

    Attributes
    ----------
    t : ndarray, shape (n_points,)
        Time points returned by the open loop solver.
    x : ndarray, shape (n_states, n_points)
        Values of the optimal state trajectory at times `t`.
    u : ndarray, shape (n_controls, n_points)
        Values of the optimal control at times `t`.
    p : ndarray, shape (n_states, n_points)
        Values of the costate at times `t`.
    v : ndarray, shape (n_points,)
        The value function evaluated at `v(x(t))`.
    """
    def __init__(self, t, x, u, p, v, **kwargs):
        self.t = t
        self.x = x
        self.u = u
        self.p = p
        self.v = v

    def __call__(self, t):
        """
        Interpolate the optimal solution at new times `t`.

        Parameters
        ----------
        t : array_like, shape (n_points,)
            Time points at which to evaluate the continuous solution.

        Returns
        -------
        x : ndarray, shape (n_states, n_points)
            Values of the optimal state trajectory at times `t`.
        u : ndarray, shape (n_controls, n_points)
            Values of the optimal control at times `t`.
        p : ndarray, shape (n_states, n_points)
            Values of the costate at times `t`.
        v : ndarray, shape (n_points,)
            The value function evaluated at `v(x(t))`.
        """
        raise NotImplementedError
