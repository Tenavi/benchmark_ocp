"""
Problem adapted from ref. [1].

##### References

1. F. Fahroo and I. M. Ross, *Pseudospectral methods for infinite-horizon
    nonlinear optimal control problems*, Journal of Guidance, Control, and
    Dynamics, 31 (2008), pp. 927-936. https://doi.org/10.2514/1.33117
"""

import numpy as np

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import resize_vector
from optimalcontrol.sampling import UniformSampler

from examples.common_utilities.dynamics import (cross_product_matrix,
                                                euler_to_quaternion)


class AttitudeControl(OptimalControlProblem):
    _required_parameters = {'J': np.diag([5., 5.1, 2.]),
                            'Wq': 1., 'Ww': 10., 'Wu': 50.,
                            'final_attitude': [0., 0., 0.],
                            'initial_max_attitude': [np.pi, np.pi/2., np.pi],
                            'initial_max_rate': np.deg2rad(5.),
                            'attitude_sample_norm': np.inf,
                            'rate_sample_norm': 2}
    _optional_parameters = {'u_lb': -0.02, 'u_ub': 0.02,
                            'attitude_sample_seed': None,
                            'rate_sample_seed': None}

    @property
    def n_states(self):
        return 7

    @property
    def n_controls(self):
        return 3

    @property
    def final_time(self):
        return np.inf

    @property
    def state_lb(self):
        """(`n_states`,) array. Lower bounds on quaternion states, specifying
        that the scalar quaternion must be positive. WARNING: currently ignored
        by `indirect` open loop solvers."""
        x_lb = np.full(self.n_states, -np.inf)
        x_lb[3] = 0.
        return x_lb

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'final_attitude' in new_params:
            try:
                obj.final_attitude = np.abs(obj.final_attitude).reshape(3)
            except ValueError:
                raise ValueError("final_attitude must be a (3,) array")
            obj._q_final = euler_to_quaternion(obj.final_attitude)
            obj._q_final = obj._q_final[:-1].reshape(3, 1)

        if 'initial_max_attitude' in new_params:
            try:
                obj.initial_max_attitude = np.reshape(
                    np.abs(obj.initial_max_attitude), (3,))
            except ValueError:
                raise ValueError("initial_max_attitude must be a (3,) array")

        if 'J' in new_params:
            try:
                obj.J = np.reshape(obj.J, (3, 3))
            except ValueError:
                raise ValueError("Inertia matrix J must be (3, 3)")
            obj._Jinv = np.linalg.inv(obj.J)
            obj._B = np.vstack((np.zeros((4, 3)), -obj._Jinv))

        for key in ('Wq', 'Ww', 'Wu'):
            if key in new_params:
                try:
                    val = float(getattr(obj, key))
                    if val <= 0.:
                        raise TypeError
                    setattr(obj, key, val)
                except TypeError:
                    raise ValueError(f"{key:s} must be a positive float")

        if not hasattr(obj, '_a0_sampler'):
            obj._a0_sampler = UniformSampler(
                lb=-obj.initial_max_attitude, ub=obj.initial_max_attitude,
                xf=obj.final_attitude,
                norm=getattr(obj, 'attitude_sample_norm'),
                seed=getattr(obj, 'attitude_sample_seed', None))
        elif ('attitude_sample_seed' in new_params
              or 'initial_max_attitude' in new_params
              or 'final_attitude' in new_params):
            obj._a0_sampler.update(
                lb=-obj.initial_max_attitude, ub=obj.initial_max_attitude,
                xf=obj.final_attitude,
                seed=new_params.get('attitude_sample_seed'))

        w0_ub = resize_vector(np.abs(obj.initial_max_rate), 3)
        if not hasattr(obj, '_w0_sampler'):
            obj._w0_sampler = UniformSampler(
                lb=-w0_ub, ub=w0_ub, xf=np.zeros(3),
                norm=getattr(obj, 'rate_sample_norm'),
                seed=getattr(obj, 'rate_sample_seed', None))
        elif ('initial_max_rate' in new_params
              or 'rate_sample_seed' in new_params):
            obj._w0_sampler.update(
                lb=-w0_ub, ub=w0_ub, seed=new_params.get('rate_sample_seed'))

    @staticmethod
    def _break_state(x):
        """
        Break up the state vector, `x=[q, q0, w]`, into individual pieces.

        Parameters
        ----------
        x : (7,) or (7, n_points) array
            States arranged by dimension, time.

        Returns
        -------
        q : (3,) or (3, n_points) array
            Vector component of quaternion, `q = x[:3]`.
        q0 : (1,) or (1, n_points) array
            Scalar component of quaternion, `q0 = x[3:4]`.
        w : (3,) or (3, n_points) array
            Angular momenta, `w = x[4:7]`.
        """
        x = np.asarray(x)
        return x[:3], x[3:4], x[4:7]

    def sample_initial_conditions(self, n_samples=1, attitude_distance=None,
                                  rate_distance=None):
        """
        Generate initial conditions. Euler angles yaw, pitch, roll are sampled
        uniformly from hypercubes defined by
        `self.parameters.initial_max_attitude`, then converted to
        quaternions. Angular rates are sampled uniformly
        from a hypercube defined by `self.parameters.initial_max_rate`.
        Optionally, one or both of these may be sampled a specified distance
        from equilibrium.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        attitude_distance : positive float, optional
            Desired distance of euler angles from
            `self.parameters.final_attitude`, in radians. The type of norm is
            determined by `self.parameters.attitude_sample_norm`. Note that
            depending on how `distance` is specified, samples may be outside the
            hypercube defined by `self.parameters.initial_max_attitude`.
        rate_distance : positive float, optional
            Desired distance of angular rates from zero, in radians/s. The type
            of norm is determined by `self.parameters.rate_sample_norm`. Note
            that depending on how `distance` is specified, samples may be
            outside the hypercube defined by `self.parameters.initial_max_rate`.

        Returns
        -------
        x0 : (7, n_samples) or (7,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a 1d array.
        """
        # Sample Euler angles in radians and convert to quaternions
        angles = self.parameters._a0_sampler(n_samples=n_samples,
                                             distance=attitude_distance)
        q = euler_to_quaternion(angles)
        # Set scalar quaternion positive
        q[-1] = np.abs(q[-1])

        # Sample angular rates in radians/s
        w = self.parameters._w0_sampler(n_samples=n_samples,
                                        distance=rate_distance)

        return np.concatenate((q, w), axis=0)

    def running_cost(self, x, u):
        q, _, w = self._break_state(x)

        if np.ndim(x) < 2:
            q_err = q - self.parameters._q_final[:, 0]
        else:
            q_err = q - self.parameters._q_final

        Lq = (self.parameters.Wq / 2.) * np.sum(q_err ** 2, axis=0)
        Lw = (self.parameters.Ww / 2.) * np.sum(w ** 2, axis=0)
        Lu = (self.parameters.Wu / 2.) * np.sum(u ** 2, axis=0)

        return Lq + Lw + Lu

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            q, q0, w = self._break_state(x)
            q_err = q - self.parameters._q_final
            dLdx = np.concatenate((self.parameters.Wq * q_err,
                                   np.zeros_like(q0),
                                   self.parameters.Ww * w))
            if squeeze:
                dLdx = dLdx[..., 0]
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self.parameters.Wu * u

            if squeeze:
                dLdu = dLdu[..., 0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            Q = np.diag(np.concatenate((np.full(3, self.parameters.Wq / 2.),
                                        [0.],
                                        np.full(3, self.parameters.Ww / 2.))))
            if not squeeze:
                Q = Q[..., None]
            if x.shape[1] > 1:
                Q = np.tile(Q, (1, 1, x.shape[1]))

            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.diag(np.full(3, self.parameters.Wu / 2.))[..., None]
            if u.shape[1] > 1:
                R = np.tile(R, (1, 1, np.shape(u)[1]))

            if squeeze:
                R = R[..., 0]
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        q, q0, w = self._break_state(x)
        u = np.reshape(u, w.shape)

        Jw = np.matmul(self.parameters.J, w)

        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + u
        dwdt = np.matmul(-self.parameters._Jinv, dwdt)

        return np.concatenate((dqdt, dq0dt, dwdt), axis=0)

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dfdx:
            q, q0, w = self._break_state(x)

            Jw = np.matmul(self.parameters.J, w)

            wx = cross_product_matrix(w)
            qx = cross_product_matrix(q)
            Jwx = cross_product_matrix(Jw)

            q0_diag = np.kron(np.eye(3), q0).reshape(3, 3, -1)

            dfdx = np.zeros((self.n_states, *x.shape))

            dfdx[3, :3] = -0.5 * w
            dfdx[3, 4:] = -0.5 * q
            dfdx[:3, 3] = 0.5 * w
            dfdx[:3, :3] = -0.5 * wx
            dfdx[:3, 4:] = 0.5 * (qx + q0_diag)
            dfdx[4:, 4:] = np.matmul(
                Jwx.T - np.matmul(self.parameters.J.T, wx.T),
                self.parameters._Jinv.T).T

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
        u = np.matmul(self.parameters._Jinv.T, p[4:]) / self.parameters.Wu
        return self._saturate(u)

    def hamiltonian_minimizer_jac(self, x, p, u0=None):
        return np.zeros((self.n_controls, *np.shape(x)))

    def bvp_dynamics(self, t, xp):
        # Extract states and costates
        x = xp[:7]
        p = xp[7:14]

        q, q0, w = self._break_state(x)
        p_q, p_q0, p_w = self._break_state(p)

        if np.ndim(x) < 2:
            q_err = q - self.parameters._q_final[:, 0]
        else:
            q_err = q - self.parameters._q_final

        u = self.hamiltonian_minimizer(x, p)
        L = self.running_cost(x, u)

        Jw = np.matmul(self.parameters.J, w)
        Jpw = np.matmul(self.parameters._Jinv.T, p_w)

        # State dynamics
        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + u
        dwdt = np.matmul(-self.parameters._Jinv, dwdt)

        # Costate dynamics
        dpq0dt = - 0.5 * np.sum(w * p_q, axis=0, keepdims=True)

        dpqdt = ((- self.parameters.Wq) * q_err
                 + 0.5 * (- np.cross(w, p_q, axis=0) + p_q0 * w))

        dpwdt = (- self.parameters.Ww * w
                 + 0.5 * (np.cross(q, p_q, axis=0) - q0 * p_q + p_q0 * q)
                 + np.matmul(-self.parameters.J.T, np.cross(w, Jpw, axis=0))
                 + np.cross(Jw, Jpw, axis=0))

        L = -L.reshape(dq0dt.shape)

        return np.concatenate((dqdt, dq0dt, dwdt, dpqdt, dpq0dt, dpwdt, L),
                              axis=0)
