import numpy as np
from scipy.interpolate import BarycentricInterpolator, make_interp_spline

from optimalcontrol.utilities import saturate
from optimalcontrol.open_loop.solutions import OpenLoopSolution
from .radau import TimeMapLog2
from .setup_nlp import separate_vars


class DirectSolution(OpenLoopSolution):
    def __init__(self, t, x, u, p, v, status, message, time_map=TimeMapLog2,
                 time_scale=1., tau=None, u_lb=None, u_ub=None):
        self._u_lb, self._u_ub = u_lb, u_ub

        if tau is None:
            tau = time_map.physical_to_radau(t * time_scale)

        self._time_map = time_map
        self._time_scale = float(time_scale)

        # BarycentricInterpolator uses np.random.permutation at initialization,
        # so control the random behavior
        np.random.seed(1234)

        self._x_interp = BarycentricInterpolator(tau, x, axis=-1)
        self._u_interp = BarycentricInterpolator(tau, u, axis=-1)
        self._p_interp = BarycentricInterpolator(tau, p, axis=-1)
        self._v_interp = make_interp_spline(t, v, k=1)

        np.random.seed()

        super().__init__(t, x, u, p, v, status, message)

    def __call__(self, t, return_x=True, return_u=True, return_p=True,
                 return_v=True):
        tau = self._time_map.physical_to_radau(t * self._time_scale)

        if return_x:
            x = self._x_interp(tau)

        if return_u:
            u = self._u_interp(tau)
            u = saturate(u, self._u_lb, self._u_ub)

        if return_p:
            p = self._p_interp(tau)

        if return_v:
            v = self._v_interp(t)

        return self._get_return_args(x=x if return_x else None,
                                     u=u if return_u else None,
                                     p=p if return_p else None,
                                     v=v if return_v else None)

    @classmethod
    def from_minimize_result(cls, minimize_result, ocp, tau, w,
                             time_map=TimeMapLog2, time_scale=1., order='F'):
        """
        Instantiate a `DirectSolution` from the result of calling
        `scipy.optimize.minimize` on a nonlinear programming problem set up with
        `.solve.solve_infinite_horizon`

        Parameters
        ----------
        minimize_result : `OptimizeResult`
            Solution of the nonlinear programming problem generated by
            pseudospectral collocation of `ocp`, as returned by
            `scipy.optimize.minimize`.
        ocp : `OptimalControlProblem`
            The optimal control problem which was solved.
        tau : (n_nodes,) array
            LGR collocation nodes on [-1, 1). See `.radau.make_lgr`.
        w : (n_nodes,) array
            LGR quadrature weights corresponding to the collocation points
            `tau`. See `.radau.make_lgr`.
        time_map : `TimeMapRadau`, default=`TimeMapLog2`
            `TimeMapRadau` subclass implementing `physical_to_radau` and
            `radau_to_physical` methods.
        time_scale : float, default=1
            Time scaling constant. It is assumed that the problem was solved
            with dynamics that were re-parameterized in terms of
            `s = time_scale * t`.
        order : {'C', 'F'}, default='F'
            If the problem was set up with C ('C', row-major) or Fortran
            ('F', column-major) ordering.

        Returns
        -------
        ocp_sol : `DirectSolution`
            The open-loop solution to `ocp` extracted from `minimize_result`.
        """
        t = time_map.radau_to_physical(tau) / time_scale
        x, u = separate_vars(minimize_result.x, ocp.n_states, ocp.n_controls,
                             order=order)

        # Extract KKT multipliers and use to approximate costates
        p = minimize_result.kkt['eq'][0].reshape(x.shape, order=order)
        p = - p / w.reshape(1, -1)

        L = ocp.running_cost(x, u)
        v = np.empty_like(w)
        for k in range(v.shape[0]):
            v[k] = np.matmul(L[k:], w[k:])

        return cls(t, x, u, p, v,
                   minimize_result.status, minimize_result.message,
                   time_map=time_map, time_scale=time_scale, tau=tau,
                   u_lb=ocp.control_lb, u_ub=ocp.control_ub)
