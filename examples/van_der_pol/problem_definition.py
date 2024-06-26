import numpy as np

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.sampling import UniformSampler


class VanDerPol(OptimalControlProblem):
    _required_parameters = {'Wx': 1, 'Wy': 1., 'Wu': 4., 'xf': 0.,
                            'mu': 2., 'b': 1.,
                            'x0_ub': np.array([[3.], [3.]]),
                            'x0_lb': -np.array([[3.], [3.]])}
    _optional_parameters = {'u_lb': -1., 'u_ub': 1., 'x0_sample_seed': None,
                            'x0_sample_norm': np.inf}

    @property
    def n_states(self):
        return 2

    @property
    def n_controls(self):
        return 1

    @property
    def final_time(self):
        return np.inf

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'xf' in new_params:
            if np.size(obj.xf) != 1:
                raise ValueError("xf must be a float")
            obj.xf = np.array([np.reshape(obj.xf, (1,)), [0.]])

        if 'b' in new_params:
            if np.size(obj.b) != 1:
                raise ValueError("b must be a float")
            obj._B = np.zeros((2, 1))
            obj._B[1] = obj.b

        if 'b' in new_params or 'xf' in new_params:
            obj.uf = float(np.squeeze(obj.xf[0] / obj.b))

        if not hasattr(obj, '_x0_sampler'):
            obj._x0_sampler = UniformSampler(
                lb=obj.x0_lb, ub=obj.x0_ub, xf=obj.xf,
                norm=getattr(obj, 'x0_sample_norm', np.inf),
                seed=getattr(obj, 'x0_sample_seed', None))
        elif any(['x0_lb' in new_params, 'x0_ub' in new_params,
                  'x0_sample_seed' in new_params, 'xf' in new_params]):
            obj._x0_sampler.update(
                lb=new_params.get('x0_lb'), ub=new_params.get('x0_ub'),
                xf=obj.xf, seed=new_params.get('x0_sample_seed'))

    def sample_initial_conditions(self, n_samples=1, distance=None):
        """
        Generate initial conditions uniformly from a rectangle, or optionally
        with a specified distance from equilibrium.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance of samples from `self.parameters.xf`. The type of
            norm is determined by `self.parameters.x0_sample_norm`. Note that
            depending on how `distance` is specified, samples may be outside the
            rectangle defined by `self.parameters.x0_lb` and
            `self.parameters.x0_ub`.

        Returns
        -------
        x0 : (2, n_samples) or (2,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a 1d array.
        """
        return self.parameters._x0_sampler(n_samples=n_samples,
                                           distance=distance)

    def running_cost(self, x, u):
        x_err, u_err, squeeze = self._center_inputs(x, u, self.parameters.xf,
                                                    self.parameters.uf)
        x_err = x_err ** 2
        u_err = u_err ** 2

        L = (self.parameters.Wx/2.) * x_err[:1]
        L += (self.parameters.Wy/2.) * x_err[1:]
        L += (self.parameters.Wu/2.) * u_err

        if squeeze:
            return L[0, 0]

        return L[0]

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x_err, u_err, squeeze = self._center_inputs(x, u, self.parameters.xf,
                                                    self.parameters.uf)

        if return_dLdx:
            x1 = x_err[:1]
            x2 = x_err[1:]

            dLdx = np.concatenate((self.parameters.Wx * x1,
                                   self.parameters.Wy * x2))

            if squeeze:
                dLdx = dLdx[..., 0]

            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self.parameters.Wu * u_err

            if squeeze:
                dLdu = dLdu[..., 0]

            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        if return_dLdx:
            Q = np.diag([self.parameters.Wx / 2., self.parameters.Wy / 2.])
            if np.ndim(x) >= 2:
                Q = np.tile(Q[..., None], (1, 1, np.shape(x)[1]))
            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.reshape(self.parameters.Wu / 2., (1, 1, 1))
            if np.ndim(u) > 1 and np.shape(u)[1] > 1:
                R = np.tile(R, (1, 1, np.shape(u)[1]))

            if np.ndim(u) < 2:
                R = R[..., 0]
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        x = np.asarray(x)
        u = np.asarray(u)

        if x.ndim < 2 and u.ndim > 1:
            u = u[:, 0]

        x1 = x[:1]
        x2 = x[1:]

        dx1dt = x2
        dx2dt = (self.parameters.mu * (1. - x1**2) * x2 - x1
                 + self.parameters.b * u)

        return np.concatenate((dx1dt, dx2dt), axis=0)

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dfdx:
            dfdx = np.zeros((self.n_states, *x.shape))
            dfdx[0, 1] = 1.
            dfdx[1, 0] = -1. - 2. * self.parameters.mu * x[0] * x[1]
            dfdx[1, 1] = self.parameters.mu * (1. - x[0]**2)

            if squeeze:
                dfdx = dfdx[..., 0]
            if not return_dfdu:
                return dfdx

        if return_dfdu:
            dfdu = np.tile(self.parameters._B[..., None], (1, 1, u.shape[1]))

            if squeeze:
                dfdu = dfdu[..., 0]
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def hamiltonian_minimizer(self, x, p):
        u = self.parameters.uf - self.parameters.b / self.parameters.Wu * p[1:]
        return self._saturate(u)

    def hamiltonian_minimizer_jac(self, x, p, u0=None):
        return np.zeros((self.n_controls, *np.shape(x)))

    def bvp_dynamics(self, t, xp):
        u = self.hamiltonian_minimizer(xp[:2], xp[2:4])
        L = self.running_cost(xp[:2], u)

        # Extract states and costates
        x1 = xp[:1]
        x2 = xp[1:2]
        x1_err = x1 - self.parameters.xf[:1]

        p1 = xp[2:3]
        p2 = xp[3:4]

        # State dynamics
        dx1dt = x2
        dx2dt = (self.parameters.mu * (1. - x1**2) * x2 - x1
                 + self.parameters.b * u)

        # Costate dynamics
        dp1dt = (-self.parameters.Wx * x1_err
                 + p2 * (2.*self.parameters.mu * x1 * x2 + 1.))
        dp2dt = (-self.parameters.Wy * x2 - p1
                 - p2 * self.parameters.mu * (1. - x1**2))

        dxpdt = np.vstack((dx1dt, dx2dt, dp1dt, dp2dt, -L))

        if u.ndim < 2:
            return dxpdt[:, 0]

        return dxpdt
