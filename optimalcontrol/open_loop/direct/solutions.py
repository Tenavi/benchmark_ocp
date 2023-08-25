from scipy.interpolate import BarycentricInterpolator, interp1d

from .legendre_gauss_radau import time_map, invert_time_map
from optimalcontrol.utilities import saturate
from optimalcontrol.open_loop.solutions import OpenLoopSolution


class DirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, status, message, tau=None, u_lb=None,
                 u_ub=None):
        self._u_lb, self._u_ub = u_lb, u_ub

        if tau is None:
            tau = time_map(t)

        self._x_interp = BarycentricInterpolator(tau, x)
        self._u_interp = BarycentricInterpolator(tau, u)
        self._p_interp = BarycentricInterpolator(tau, p)
        self._v_interp = interp1d(t, v)

        super().__init__(t, x, u, p, v, status, message)

    def __call__(self, t):
        tau = time_map(t)

        x = self._x_interp(tau)
        u = self._u_interp(tau)
        u = saturate(u, self._u_lb, self._u_ub)
        p = self._p_interp(tau)

        v = self._v_interp(t)

        return x, u, p, v

    @classmethod
    def from_minimize_result(cls, minimize_result, tau, w, order, separate_vars,
                             total_cost, u_ub=None, u_lb=None):
        t = invert_time_map(tau)
        x, u = separate_vars(minimize_result.x)

        # Extract KKT multipliers and use to approximate costates
        p = minimize_result.kkt['eq'][0].reshape(x.shape, order=order)
        p = p / w.reshape(1, -1)

        v = total_cost(t, x, u)

        return cls(t, x, u, p, v, minimize_result.status,
                   minimize_result.message, tau=tau, u_lb=u_lb, u_ub=u_ub)
