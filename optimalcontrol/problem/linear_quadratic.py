import numpy as np
from scipy.spatial.distance import cdist

from ..sampling import UniformSampler
from ..utilities import saturate, find_saturated, resize_vector
from .problem import OptimalControlProblem


class LinearQuadraticProblem(OptimalControlProblem):
    """
    General class for defining infinite horizon linear quadratic regulator
    problems. Takes the following parameters upon initialization.

    Parameters
    ----------
    A : (n_states, n_states) array
        State Jacobian matrix, $df/dx (x_f, u_f)$.
    B : (n_states, n_controls) array
        Control Jacobian matrix, $df/du (x_f, u_f)$.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states,
        $d^2L/dx^2 (x_f, u_f)$. Must be positive semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls,
        $d^2L/du^2 (x_f, u_f)$. Must be positive definite.
    xf : {(n_states, 1) array, float}, default=0.
        Goal state, nominal linearization point. If float, will be broadcast
        into an array of shape `(n_states, 1)`.
    uf : {(n_controls, 1) array, float}, default=0.
        Control values at nominal linearization point. If float, will be
        broadcast into an array of shape `(n_controls, 1)`.
    x0_lb : {(n_states, 1) array, float}
        Lower bounds for hypercube from which to sample initial conditions `x0`.
        If float, will be broadcast into an array of shape `(n_states, 1)`.
    x0_ub : {(n_states, 1) array, float}
        Upper bounds for hypercube from which to sample initial conditions `x0`.
        If float, will be broadcast into an array of shape `(n_states, 1)`.
    u_lb : {(n_controls, 1) array, float}, optional
        Lower control bounds. If float, will be broadcast into an array of shape
        `(n_controls, 1)`.
    u_ub : {(n_controls, 1) array, float}, optional
        Upper control bounds. If float, will be broadcast into an array of shape
        `(n_controls, 1)`.
    x0_sample_seed : int, optional
        Random seed to use for sampling initial conditions.
    """
    _required_parameters = {'A': None, 'B': None, 'Q': None, 'R': None,
                            'xf': 0., 'uf': 0., 'x0_lb': None, 'x0_ub': None}
    _optional_parameters = {'u_lb': None, 'u_ub': None, 'x0_sample_seed': None}

    def _saturate(self, u):
        return saturate(u, self.parameters.u_lb, self.parameters.u_ub)

    @property
    def n_states(self):
        return self.parameters.n_states

    @property
    def n_controls(self):
        return self.parameters.n_controls

    @property
    def final_time(self):
        return np.inf

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'A' in new_params:
            try:
                obj.A = np.atleast_1d(obj.A)
                obj.n_states = obj.A.shape[0]
                obj.A = obj.A.reshape(obj.n_states, obj.n_states)
            except ValueError:
                raise ValueError("State Jacobian matrix A must have shape "
                                 "(n_states, n_states)")

        if 'B' in new_params:
            try:
                obj.B = np.asarray(obj.B)
                if obj.B.ndim == 2 and obj.B.shape[0] != obj.n_states:
                    raise ValueError
                obj.B = np.reshape(obj.B, (obj.n_states, -1))
                obj.n_controls = obj.B.shape[1]
            except ValueError:
                raise ValueError("Control Jacobian matrix B must have shape "
                                 "(n_states, n_controls)")

        if 'Q' in new_params:
            try:
                obj.Q = np.reshape(obj.Q, (obj.n_states, obj.n_states))
                eigs = np.linalg.eigvals(obj.Q)
                if not np.all(eigs >= 0.) or not np.allclose(obj.Q, obj.Q.T):
                    raise ValueError
                obj.singular_Q = np.any(np.isclose(eigs, 0.))
            except ValueError:
                raise ValueError("State cost matrix Q must have shape "
                                 "(n_states, n_states) and be positive "
                                 "semi-definite")

        if 'R' in new_params:
            try:
                obj.R = np.reshape(obj.R, (obj.n_controls, obj.n_controls))
                eigs = np.linalg.eigvals(obj.R)
                if not np.all(eigs > 0.) or not np.allclose(obj.R, obj.R.T):
                    raise ValueError
            except ValueError:
                raise ValueError("Control cost matrix R must have shape "
                                 "(n_controls, n_controls) and be positive "
                                 "definite")

        if 'xf' in new_params:
            obj.xf = resize_vector(obj.xf, obj.n_states)

        if 'uf' in new_params:
            obj.uf = resize_vector(obj.uf, obj.n_controls)

        for key in ('u_lb', 'u_ub'):
            if key in new_params:
                if getattr(obj, key, None) is not None:
                    u_bound = resize_vector(new_params[key], obj.n_controls)
                    setattr(obj, key, u_bound)

        if 'B' in new_params or 'R' in new_params:
            obj._RB2 = np.linalg.solve(obj.R, obj.B.T) / 2.

        if 'Q' in new_params or not hasattr(obj, '_x0_sampler'):
            obj._x0_sampler = UniformSampler(
                lb=obj.x0_lb, ub=obj.x0_ub, xf=obj.xf,
                norm=2 if obj.singular_Q else obj.Q,
                seed=getattr(obj, 'x0_sample_seed', None))
        elif any(['x0_lb' in new_params, 'x0_ub' in new_params,
                  'x0_sample_seed' in new_params, 'xf' in new_params]):
            obj._x0_sampler.update(
                lb=new_params.get('x0_lb'), ub=new_params.get('x0_ub'),
                xf=obj.xf, seed=new_params.get('x0_sample_seed'))

    def sample_initial_conditions(self, n_samples=1, distance=None):
        """
        Generate initial conditions uniformly from a hypercube, or on the
        surface of a hyper-ellipse defined by `self.parameters.Q`.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance of samples from `self.parameters.xf`. If
            `self.parameters.Q` is positive definite, the distance is defined by
            the norm `norm(x) = sqrt(x.T @ Q @ x)`, otherwise uses the l2 norm.

        Returns
        -------
        x0 : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a one-dimensional array.
        """
        return self.parameters._x0_sampler(n_samples=n_samples,
                                           distance=distance)

    def distances(self, xa, xb):
        """
        Calculate the distance of a batch of states from another state or batch
        of states. The distance is defined as
        `distances(xa - xb) = sqrt((xa - xb).T @ Q @ (xa - xb))`.

        Parameters
        ----------
        xa : (n_states, n_a) or (n_states,) array
            First batch of points.
        xb : (n_states, n_b) or (n_states,) array
            Second batch of points.

        Returns
        -------
        dist : (n_a, n_b) array
            `sqrt((xa - xb).T @ Q @ (xa - xb))` for each point (column) in `xa`
            and `xb`.
        """
        xa = np.reshape(xa, (self.n_states, -1)).T
        xb = np.reshape(xb, (self.n_states, -1)).T

        return cdist(xa, xb, metric='mahalanobis', VI=self.parameters.Q)

    def running_cost(self, x, u):
        x_err, u_err, squeeze = self._center_inputs(x, u, self.parameters.xf,
                                                    self.parameters.uf)

        # Batch multiply (x - xf).T @ Q @ (x - xf)
        L = np.einsum('ij,ij->j', x_err, self.parameters.Q @ x_err)

        # Batch multiply (u - uf).T @ R @ (u - xf) and sum
        L += np.einsum('ij,ij->j', u_err, self.parameters.R @ u_err)

        if squeeze:
            return L[0]

        return L

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x_err, u_err, squeeze = self._center_inputs(x, u, self.parameters.xf,
                                                    self.parameters.uf)

        if return_dLdx:
            dLdx = 2. * np.einsum('ij,jb->ib', self.parameters.Q, x_err)
            if squeeze:
                dLdx = dLdx[..., 0]
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = 2. * np.einsum('ij,jb->ib', self.parameters.R, u_err)

            # Where the control is saturated, the gradient is zero
            sat_idx = find_saturated(np.reshape(u, dLdu.shape),
                                     self.parameters.u_lb, self.parameters.u_ub)
            dLdu[sat_idx] = 0.

            if squeeze:
                dLdu = dLdu[..., 0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            dLdx = np.copy(self.parameters.Q)
            if not squeeze:
                dLdx = np.tile(dLdx[..., None], (1, 1, x.shape[1]))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = np.tile(self.parameters.R[..., None], (1, 1, u.shape[1]))

            # Where the control is saturated, the gradient is zero (constant).
            # This makes the Hessian zero in all terms that include a saturated
            # control
            sat_idx = find_saturated(u, self.parameters.u_lb,
                                     self.parameters.u_ub)
            sat_idx = sat_idx[None, ...] + sat_idx[:, None, :]
            dLdu[sat_idx] = 0.

            if squeeze:
                dLdu = dLdu[..., 0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def dynamics(self, x, u):
        x, u, squeeze = self._center_inputs(x, u, self.parameters.xf,
                                            self.parameters.uf)

        dxdt = np.matmul(self.parameters.A, x) + np.matmul(self.parameters.B, u)

        if squeeze:
            return dxdt.flatten()

        return dxdt

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dfdx:
            dfdx = np.copy(self.parameters.A)
            if not squeeze:
                dfdx = np.tile(dfdx[..., None], (1, 1, x.shape[1]))
            if not return_dfdu:
                return dfdx

        if return_dfdu:
            dfdu = np.tile(self.parameters.B[..., None], (1, 1, u.shape[1]))

            # Where the control is saturated, the Jacobian is zero
            idx = find_saturated(u, self.parameters.u_lb, self.parameters.u_ub)
            dfdu[:, idx] = 0.

            if squeeze:
                dfdu = dfdu[..., 0]
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        p = np.reshape(p, (self.n_states, -1))
        u = self.parameters.uf - np.matmul(self.parameters._RB2, p)
        u = self._saturate(u)

        if np.ndim(x) < 2:
            return u.flatten()

        return u

    def optimal_control_jac(self, x, p, u0=None):
        return np.zeros((self.n_controls, self.n_states) + np.shape(p)[1:])
