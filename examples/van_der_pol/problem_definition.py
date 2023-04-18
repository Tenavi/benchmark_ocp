import numpy as np

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import saturate, resize_vector
from optimalcontrol.sampling import UniformSampler


class VanDerPol(OptimalControlProblem):
    _required_parameters = {'Wx': .5, 'Wy': 1., 'Wu': 4., 'xf': 0.,
                            'mu': 2., 'b': 1.5,
                            'x0_ub': np.array([[3.], [4.]]),
                            'x0_lb': -np.array([[3.], [4.]])}
    _optional_parameters = {'u_lb': -1., 'u_ub': 1., 'x0_sample_seed': None,
                            'x0_sample_norm': np.inf}

    def _saturate(self, u):
        return saturate(u, self._params.u_lb, self._params.u_ub)

    @property
    def n_states(self):
        return 2

    @property
    def n_controls(self):
        return 1

    @property
    def final_time(self):
        return np.inf

    def _update_params(self, obj, **new_params):
        if 'xf' in new_params:
            self.xf = np.zeros((2, 1))
            self.xf[0] = obj.xf

        if 'b' in new_params:
            self.B = np.zeros((2, 1))
            self.B[1] = obj.b

        if 'b' in new_params or 'xf' in new_params:
            self._uf = float(self.xf[0] / obj.b)
            self.uf = np.reshape(self._uf, (1, 1))

        for key in ('u_lb', 'u_ub'):
            if key in new_params:
                u_bound = resize_vector(new_params[key], self.n_controls)
            else:
                u_bound = getattr(obj, key, None)
            setattr(obj, key, u_bound)

        if not hasattr(self, '_x0_sampler'):
            self._x0_sampler = UniformSampler(
                lb=obj.x0_lb, ub=obj.x0_ub, xf=self.xf,
                norm=getattr(obj, 'x0_sample_norm', 1),
                seed=getattr(obj, 'x0_sample_seed', None))
        elif any(['x0_lb' in new_params, 'x0_ub' in new_params,
                  'x0_sample_seed' in new_params, 'xf' in new_params]):
            self._x0_sampler.update(
                lb=new_params.get('x0_lb'), ub=new_params.get('x0_ub'),
                xf=self.xf, seed=new_params.get('x0_sample_seed'))

    def sample_initial_conditions(self, n_samples=1, distance=None):
        """
        Generate initial conditions uniformly from a rectangle, or optionally
        with a specified distance from equilibrium.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance of samples from `self.xf`. The type of norm is
            determined by `self.parameters.x0_sample_norm`. Note that depending
            on how `distance` is specified, samples may be outside the rectangle
            defined by `self.parameters.x0_lb` and `self.parameters.x0_ub`.

        Returns
        -------
        x0 : (2, n_samples) or (2,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a 1d array.
        """
        return self._x0_sampler(n_samples=n_samples, distance=distance)

    def running_cost(self, x, u):
        if np.ndim(x) < 2:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf
        x_err = x_err ** 2

        u_err = (self._saturate(u) - self._uf) ** 2

        L = (self._params.Wx/2.) * x_err[:1]
        L += (self._params.Wy/2.) * x_err[1:]
        L += (self._params.Wu/2.) * u_err

        return L[0]

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        if np.ndim(x) < 2:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf

        u_err = self._saturate(u) - self._uf

        x1 = x_err[:1]
        x2 = x_err[1:]

        if return_dLdx:
            dLdx = np.concatenate((self._params.Wx * x1, self._params.Wy * x2))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self._params.Wu * u_err
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        if return_dLdx:
            Q = np.diag([self._params.Wx, self._params.Wy])
            if np.ndim(x) >= 2:
                Q = np.tile(Q[..., None], (1, 1, np.shape(x)[1]))
            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.reshape(self._params.Wu, (1, 1))
            if np.ndim(u) >= 2:
                R = np.tile(R[..., None], (1, 1, np.shape(u)[1]))
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        u = self._saturate(u)
        if np.ndim(x) < 2 and np.ndim(u) > 1:
            u = u.flatten()

        x1 = x[:1]
        x2 = x[1:]

        dx1dt = x2
        dx2dt = self._params.mu * (1. - x1**2) * x2 - x1 + self._params.b * u

        return np.concatenate((dx1dt, dx2dt), axis=0)

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        if return_dfdx:
            dfdx = np.zeros((self.n_states, * np.shape(x)))
            dfdx[0, 1] = 1.
            dfdx[1, 0] = -1. - 2.*self._params.mu*x[0]*x[1]
            dfdx[1, 1] = self._params.mu*(1. - x[0]**2)

            if not return_dfdu:
                return dfdx

        if return_dfdu:
            if np.ndim(u) > 1:
                dfdu = np.tile(self.B[..., None], (1, 1, np.shape(u)[-1]))
            else:
                dfdu = np.copy(self.B)

            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        u = self._uf - self._params.b / self._params.Wu * p[1:]
        return self._saturate(u)

    def optimal_control_jac(self, x, p, u0=None):
        return np.zeros((self.n_controls, self.n_states) + np.shape(p)[1:])

    def bvp_dynamics(self, t, xp):
        u = self.optimal_control(xp[:2], xp[2:-1])
        L = self.running_cost(xp[:2], u)

        x1 = xp[:1]
        x2 = xp[1:2]

        x1_err = x1 - self.xf[:1]

        # Costate
        p1 = xp[2:3]
        p2 = xp[3:4]

        # State dynamics
        dx1dt = x2
        dx2dt = self._params.mu * (1. - x1**2) * x2 - x1 + self._params.b * u

        # Costate dynamics
        dp1dt = -self._params.Wx * x1_err + p2 * (2.*self._params.mu*x1*x2 + 1.)
        dp2dt = -self._params.Wy * x2 - p1 - p2 * self._params.mu * (1. - x1**2)

        return np.vstack((dx1dt, dx2dt, dp1dt, dp2dt, -L))
