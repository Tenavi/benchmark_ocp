import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import saturate, find_saturated, resize_vector
from optimalcontrol.sampling import UniformSampler


def cross_product_matrix(omega):
    """

    Parameters
    ----------
    omega

    Returns
    -------

    """
    zeros = np.zeros_like(omega[0])
    return np.array([[zeros, -omega[2], omega[1]],
                     [omega[2], zeros, -omega[0]],
                     [-omega[1], omega[0], zeros]])


def quaternion_to_euler(quat, degrees=False, normalize=True,
                        ignore_warnings=True):
    """

    Parameters
    ----------
    quat
    degrees
    normalize
    ignore_warnings

    Returns
    -------
    angles
    """
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore', category=UserWarning)
        angles = Rotation(quat.T, normalize=normalize)
        return angles.as_euler('ZYX', degrees=degrees).T


def euler_to_quaternion(yaw, pitch, roll, degrees=False):
    """

    Parameters
    ----------
    yaw
    pitch
    roll
    degrees

    Returns
    -------
    quat
    """
    angles = np.asarray([yaw, pitch, roll], dtype=float) / 2.
    if degrees:
        angles = np.deg2rad(angles)

    cos = np.cos(angles)
    sin = np.sin(angles)

    yaw_pitch = np.vstack((cos[0] * cos[1],   # cos(yaw/2) * cos(pitch/2)
                           cos[0] * sin[1],   # cos(yaw/2) * sin(pitch/2)
                           sin[0] * cos[1],   # sin(yaw/2) * cos(pitch/2)
                           sin[0] * sin[1]))  # sin(yaw/2) * sin(pitch/2)

    roll = np.vstack((cos[-1], sin[-1]))      # cos(roll/2) * sin(roll/2)

    # Get combinations of yaw, pitch, and roll
    yaw_pitch_roll = np.einsum('ik,jk->ijk', yaw_pitch, roll)

    q = np.vstack((yaw_pitch_roll[0, 1] - yaw_pitch_roll[3, 0],
                   yaw_pitch_roll[1, 0] + yaw_pitch_roll[2, 1],
                   yaw_pitch_roll[2, 0] - yaw_pitch_roll[1, 1],
                   yaw_pitch_roll[0, 0] + yaw_pitch_roll[3, 1]))

    if angles.ndim == 1:
        return q[:, 0]

    return q


class AttitudeControl(OptimalControlProblem):
    _required_parameters = {'J': [[59.22, -1.14, -0.8],
                                  [-1.14, 40.56, 0.1],
                                  [-0.8, 0.1, 57.6]],
                            'Wq': 1., 'Ww': 1., 'Wu': 1.,
                            'final_attitude': [0., 0., 0.],
                            'initial_max_attitude': [np.pi, np.pi/2., np.pi],
                            'initial_max_rate': 0.1,
                            'attitude_sample_norm': np.inf,
                            'rate_sample_norm': 2}
    _optional_parameters = {'u_lb': -0.3, 'u_ub': 0.3,
                            'attitude_sample_seed': None,
                            'rate_sample_seed': None}

    def _saturate(self, u):
        return saturate(u, self._params.u_lb, self._params.u_ub)

    @property
    def n_states(self):
        return 7

    @property
    def n_controls(self):
        return 3

    @property
    def final_time(self):
        return np.inf

    def _update_params(self, obj, **new_params):
        if 'final_attitude' in new_params:
            try:
                obj.final_attitude = np.abs(obj.final_attitude).reshape(3)
            except:
                raise ValueError("final_attitude must be a (3,) array")
            if np.any(obj.final_attitude > np.pi):
                raise ValueError("Final pitch and roll must be between -pi and "
                                 "pi")
            if obj.final_attitude[1] > np.pi / 2.:
                raise ValueError("Final yaw must be between -pi/2 and pi/2")
            self._q_final = euler_to_quaternion(*obj.final_attitude)
            self._q_final = self._q_final[1:].reshape(3, 1)

        if 'J' in new_params:
            try:
                obj.J = np.reshape(obj.J, (3, 3))
            except:
                raise ValueError("Inertia matrix J must be (3, 3)")
            self._J = obj.J
            self._JT = self._J.T
            self._Jinv = np.linalg.inv(self._J)
            self._JinvT = self._Jinv.T
            self._B = np.vstack((np.zeros((4, 3)), -self._Jinv))

        for key in ('Wq', 'Ww', 'Wu'):
            if key in new_params:
                try:
                    val = float(getattr(obj, key))
                    assert val > 0.
                    setattr(obj, key, val)
                except:
                    raise ValueError(f"{key:s} must be a positive float")

        for key in ('u_lb', 'u_ub'):
            if key in new_params:
                u_bound = resize_vector(new_params[key], self.n_controls)
            else:
                u_bound = getattr(obj, key, None)
            setattr(obj, key, u_bound)

        if 'initial_max_attitude' in new_params:
            try:
                obj.initial_max_attitude = np.reshape(
                    np.abs(obj.initial_max_attitude), (3,))
            except:
                raise ValueError("initial_max_attitude must be a (3,) array")
            if np.any(obj.initial_max_attitude > np.pi):
                raise ValueError("Initial pitch and roll must be between -pi "
                                 "and pi")
            if obj.initial_max_attitude[1] > np.pi / 2.:
                raise ValueError("Initial yaw must be between -pi/2 and pi/2")

        if not hasattr(self, '_a0_sampler'):
            self._a0_sampler = UniformSampler(
                lb=-obj.initial_max_attitude, ub=obj.initial_max_attitude,
                xf=obj.final_attitude,
                norm=getattr(obj, 'attitude_sample_norm'),
                seed=getattr(obj, 'attitude_sample_seed', None))
        elif ('attitude_sample_seed' in new_params
              or 'initial_max_attitude' in new_params
              or 'final_attitude' in new_params):
            self._a0_sampler.update(
                lb=-obj.initial_max_attitude, ub=obj.initial_max_attitude,
                xf=obj.final_attitude,
                seed=new_params.get('attitude_sample_seed'))

        w0_ub = resize_vector(np.abs(obj.initial_max_rate), 3)
        if not hasattr(self, '_w0_sampler'):
            self._w0_sampler = UniformSampler(
                lb=-w0_ub, ub=w0_ub, xf=np.zeros(3),
                norm=getattr(obj, 'rate_sample_norm'),
                seed=getattr(obj, 'rate_sample_seed', None))
        elif ('initial_max_rate' in new_params
              or 'rate_sample_seed' in new_params):
            self._w0_sampler.update(
                lb=-w0_ub, ub=w0_ub, seed=new_params.get('rate_sample_seed'))

    @staticmethod
    def _break_state(x):
        """
        Break up the state vector, `x=[q0, q, w]`, into individual pieces.

        Parameters
        ----------
        x : (7,) or (7, n_points) array
            States arranged by dimension, time.

        Returns
        -------
        q0 : (1,) or (1, n_points) array
            Scalar component of quaternion, `q0 = x[:1]`.
        q : (4,) or (4, n_points) array
            Vector component of quaternion, `q = x[1:4]`.
        w : (3,) or (3, n_points) array
            Angular momenta, `w = x[4:7]`.
        """
        q0 = x[:1]
        q = x[1:4]
        w = x[4:7]
        return q0, q, w

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
        angles = self._a0_sampler(n_samples=n_samples,
                                  distance=attitude_distance)
        q = euler_to_quaternion(*angles)
        # Set scalar quaternion positive
        q[0] = np.abs(q[0])

        # Sample angular rates in radians/s
        w = self._w0_sampler(n_samples=n_samples, distance=rate_distance)

        return np.concatenate((q, w), axis=0)

    def running_cost(self, x, u):
        _, q, w = self._break_state(x)

        if np.ndim(x) < 2:
            q_err = q - self._q_final.flatten()
        else:
            q_err = q - self._q_final
        Lq = (self._params.Wq / 2.) * np.sum(q_err**2, axis=0)
        Lw = (self._params.Ww / 2.) * np.sum(w**2, axis=0)
        Lu = (self._params.Wu / 2.) * np.sum(self._saturate(u)**2, axis=0)

        return Lq + Lw + Lu

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            q0, q, w = self._break_state(x)
            q_err = q - self._q_final
            dLdx = np.concatenate((np.zeros_like(q0),
                                   self._params.Wq * q_err,
                                   self._params.Ww * w))
            if squeeze:
                dLdx = dLdx[..., 0]
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self._params.Wu * self._saturate(u)

            # Where the control is saturated, the gradient is zero
            sat_idx = find_saturated(u, self._params.u_lb, self._params.u_ub)
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
            Q = np.diag(np.concatenate(([0.], np.full(3, self._params.Wq / 2.),
                                        np.full(3, self._params.Ww / 2.))))
            if not squeeze:
                Q = Q[..., None]
            if x.shape[1] > 1:
                Q = np.tile(Q, (1, 1, x.shape[1]))

            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.diag(np.full(3, self._params.Wu / 2.))[..., None]
            if u.shape[1] > 1:
                R = np.tile(R, (1, 1, np.shape(u)[1]))

            # Where the control is saturated, the gradient is zero (constant).
            # This makes the Hessian zero in all terms that include a saturated
            # control
            sat_idx = find_saturated(u, self._params.u_lb, self._params.u_ub)
            sat_idx = sat_idx[None, ...] + sat_idx[:, None, ...]
            R[sat_idx] = 0.

            if squeeze:
                R = R[..., 0]
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, X, U):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        flat_out = X.ndim < 2

        q0, q, w = self._break_state(X[:7].reshape(7,-1))

        Jw = np.matmul(self.J, w)

        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + U.reshape(3,-1)
        dwdt = np.matmul(-self.Jinv, dwdt)

        dXdt = np.vstack((dq0dt, dqdt, dwdt))
        if flat_out:
            dXdt = dXdt.flatten()
        return dXdt

    def jacobians(self, X, U, F0=None):
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances. Default implementation
        approximates the Jacobians with central differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        U : (n_controls,) or (n_controls, n_points)  array
            Control inputs.
        F0 : ignored
            For API consistency only.

        Returns
        -------
        dFdX : (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        q0, q, w = self._break_state(X.reshape(7, -1))

        Jw = np.matmul(self.J, w)

        wx = cross_product_matrix(w)
        qx = cross_product_matrix(q)
        Jwx = cross_product_matrix(Jw)

        q0_diag = np.kron(np.eye(3), q0).reshape(3, 3, -1)

        dFdX = np.zeros((7, 7, w.shape[1]))
        dFdX[0,1:4] = -0.5 * w
        dFdX[0,4:] = -0.5 * q
        dFdX[1:4,0] = 0.5 * w
        dFdX[1:4,1:4] = -0.5 * wx
        dFdX[1:4,4:] = 0.5 * (qx + q0_diag)
        dFdX[4:,4:] = np.matmul(Jwx.T - np.matmul(self.JT, wx.T), self.JinvT).T

        dFdU = np.expand_dims(self.B, -1)
        dFdU = np.tile(dFdU, (1,1,q0.shape[-1]))

        return dFdX, dFdU

    def U_star(self, X, dVdX):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        U = np.matmul(self.JinvT, dVdX[4:]) / self.Wu

        return saturate_np(U, self.U_lb, self.U_ub)

    def make_U_NN(self, X, dVdX):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        from tensorflow import matmul

        U = matmul(self.Jinv.astype(np.float32) / self.Wu, dVdX[4:])

        return saturate_tf(U, self.U_lb, self.U_ub)

    def jac_U_star(self, X, dVdX, U0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : ignored
            For API consistency only.

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        dVdX = dVdX.reshape(self.n_states, -1)
        return np.zeros((self.n_controls, self.n_states, dVdX.shape[-1]))

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

        Parameters
        ----------
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        # Optimal control as a function of the costate
        U = self.U_star(X_aug[:7], X_aug[7:14])

        q0, q, w, A0, Aq, Aw = self._break_state(X_aug)

        Jw = np.matmul(self.J, w)
        JAw = np.matmul(self.JinvT, Aw)

        # State dynamics
        dq0dt = - 0.5 * np.sum(w * q, axis=0, keepdims=True)
        dqdt = 0.5 * (-np.cross(w, q, axis=0) + q0 * w)

        dwdt = np.cross(w, Jw, axis=0) + U.reshape(3,-1)
        dwdt = np.matmul(-self.Jinv, dwdt)

        # Costate dynamics
        dA0dt = - 0.5 * np.sum(w * Aq, axis=0, keepdims=True)

        dAqdt = (
            self.Wq * (self.X_bar[1:4] - q)
            + 0.5 * (- np.cross(w, Aq, axis=0) + A0 * w)
        )

        dAwdt = (
            self.Ww * (self.X_bar[4:] - w)
            + 0.5 * (np.cross(q, Aq, axis=0) - q0*Aq + A0*q)
            + np.matmul(-self.JT, np.cross(w, JAw, axis=0))
            + np.cross(Jw, JAw, axis=0)
        )

        L = np.atleast_2d(self.running_cost(X_aug[:7], U))

        return np.vstack((dq0dt, dqdt, dwdt, dA0dt, dAqdt, dAwdt, -L))

    def _constraint_fun(self, X):
        '''
        A (vector-valued) function which is zero when the quaternion norm state
        constraint is satisfied.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        -------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        return 1. - np.sum(X[:4]**2, axis=0, keepdims=True)

    def _constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        JC : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        JC = -2. * X[:4]
        return np.hstack((JC.reshape(1,-1), np.zeros((1,3))))