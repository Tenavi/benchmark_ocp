import numpy as np

from scipy.integrate import OdeSolver
from scipy.integrate._ivp.rk import rk_step, RkDenseOutput


class FixedStepSolver(OdeSolver):
    """Base class for explicit fixed stepsize Runge-Kutta methods.

    Based on `scipy.integrate._ivp.rk.RungeKutta`."""
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    _min_dt = 1e-10

    def __init__(self, fun, t0, y0, t_bound, dt=None, vectorized=False,
                 **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)

        self.y_old = None
        self.f = self.fun(self.t, self.y)
        if dt is None or dt <= np.finfo(float).eps:
            raise ValueError("dt must be positive.")
        self.dt = np.maximum(self._min_dt, self._validate_stepsize(dt))

        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)

    @property
    def n_stages(self):
        return self.C.shape[0]

    def _validate_stepsize(self, dt):
        max_dt = np.abs(self.t_bound - self.t)

        # Stepping by dt accumulates small errors. If stepping by dt would get
        # almost to the end of the time interval, use a slightly larger timestep
        # which gets there exactly.
        if dt >= self._min_dt >= max_dt - dt:
            return max_dt

        return np.minimum(dt, max_dt)

    def _step_impl(self):
        self.dt = self._validate_stepsize(self.dt)

        h = self.dt * self.direction

        y_new, f_new = rk_step(self.fun, self.t, self.y, self.f, h,
                               self.A, self.B, self.C, self.K)

        self.y_old = self.y
        self.t = self.t + h
        self.y = y_new
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class Euler(FixedStepSolver):
    """Explicit Euler method with fixed timestep.

    This is a first order method. Linear interpolation is used for dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is `fun(t, y)`.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    dt : float
        Absolute time step size. Must be positive.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    """
    C = np.array([0.])
    A = np.array([[0.]])
    B = np.array([1.])
    P = np.array([[1.],
                  [0.]])


class Midpoint(FixedStepSolver):
    """Explicit midpoint method with fixed timestep.

    This is a second order method. A quadratic Hermite polynomial is used for
    dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is `fun(t, y)`.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    dt : float
        Absolute time step size. Must be positive.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    """
    C = np.array([0., 1/2])
    A = np.array([[0., 0.],
                  [1/2, 0.]])
    B = np.array([0., 1.])
    P = np.array([[1., -1.],
                  [0., 1.],
                  [0., 0.]])


class RK4(FixedStepSolver):
    """Explicit fourth order Runge-Kutta method with fixed timestep.

    This is the classic fourth order Runge-Kutta method. A cubic Hermite
    polynomial is used for dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is `fun(t, y)`.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    dt : float
        Absolute time step size. Must be positive.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    """
    C = np.array([0., 1/2, 1/2, 1.])
    A = np.array([[0., 0., 0., 0.],
                  [1/2, 0., 0., 0.],
                  [0., 1/2, 0., 0.],
                  [0., 0., 1., 0.]])
    B = np.array([1/6, 1/3, 1/3, 1/6])

    def _dense_output_impl(self):
        y_slope = (self.y - self.y_old) / (self.t - self.t_old)

        Q = np.empty((self.n, 3))

        # Constraint on linear term that dy/dt(t0) = f(t0, y0)
        Q[:, 0] = self.K[0]
        # Constraints on quadratic and cubic terms such that
        #   y(t0 + h) = y1 and dy/dt(t0 + h) = f(t0 + h, y1)
        Q[:, 1] = - (2. * self.K[0] + self.K[-1] - 3. * y_slope)
        Q[:, 2] = self.K[0] + self.K[-1] - 2. * y_slope

        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


METHODS = {'Euler': Euler, 'Midpoint': Midpoint, 'RK4': RK4}
