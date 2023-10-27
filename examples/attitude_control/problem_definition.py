import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import resize_vector
from optimalcontrol.sampling import UniformSampler


def cross_product_matrix(w):
    """
    Construct the cross product matrix or matrices from one or more vectors.

    Parameters
    ----------
    w : (3, n_vectors) or (3,) array
        Vector(s) to construct cross product matrices for.

    Returns
    -------
    w_x : (3, 3, n_vectors) or (3, 3) array
        If `w` is a 1d array, then `w_x` is a 2d array with entries
        ```
        w_x = [[ 0,    -w[2], w[1]  ],
               [ w[2], 0,     -w[0] ],
               [ -w[1], w[0], 0     ]]
        ```
        If `w` is a 2d array, then `w_x[:, :, i]` is the cross product matrix
        (see above) for `w[:, i]`.
    """
    zeros = np.zeros_like(w[0])
    return np.array([[zeros, -w[2], w[1]],
                     [w[2], zeros, -w[0]],
                     [-w[1], w[0], zeros]])


def quaternion_to_euler(quat, degrees=False, normalize=True,
                        ignore_warnings=True):
    """
    Convert angles in quaternion representation to Euler angles.

    Parameters
    ----------
    quat : (4, n_angles) or (4,) array
        Angles in quaternion representation. `quat[:3]` are assumed to contain
        the vector portion of the quaternion, and `quat[3]` is asssumed to
        contain the scalar portion.
    degrees : bool, default=False
        If `degrees=False` (default), output Euler angles in radians. If True,
        convert these to degrees.
    normalize : bool, default=True
        If `normalize=True` (default), quaternions are scaled to have unit norm
        before converting to Euler angles.
    ignore_warnings : bool, default=True
        Set `ignore_warnings=True` (default) to suppress a `UserWarning` about
        gimbal lock, if it occurs.

    Returns
    -------
    angles : (3, n_angles) or (3,) array
        `quat` converted to Euler angle representation. `angles[0]` contains
        yaw, `angles[1]` contains pitch, and `angles[2]` contains roll.
    """
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore', category=UserWarning)
        angles = Rotation(np.asarray(quat).T, normalize=normalize)
        return angles.as_euler('ZYX', degrees=degrees).T


def euler_to_quaternion(angles, degrees=False):
    """
    Convert Euler angles to quaternion representation.

    Parameters
    ----------
    angles : (3, n_angles) or (3,) array
        Euler angles to convert to quaternion representation. `angles[0]`
        is assumed to contain yaw, `angles[1]` pitch, and `angles[2]` roll.
    degrees : bool, default=False
        If `degrees=False` (default), assumes `angles` are in radians. If True,
        assumes `angles` are in degrees.

    Returns
    -------
    quat : (4, n_angles) or (4,) array
        `angles` in quaternion representation. `quat[:3]` contains the vector
        portion of the quaternion, and `quat[3]` contains the scalar portion.
    """
    angles = Rotation.from_euler('ZYX', np.asarray(angles).T, degrees=degrees)
    return angles.as_quat().T


class AttitudeControl(OptimalControlProblem):
    _required_parameters = {'J': [[59.22, -1.14, -0.8],
                                  [-1.14, 40.56, 0.1],
                                  [-0.8, 0.1, 57.6]],
                            'Wq': 1/4, 'Ww': 1/2, 'Wu': 1.,
                            'final_attitude': [0., 0., 0.],
                            'initial_max_attitude': [np.pi, np.pi/2., np.pi],
                            'initial_max_rate': 0.15,
                            'attitude_sample_norm': np.inf,
                            'rate_sample_norm': 2}
    _optional_parameters = {'u_lb': -0.2, 'u_ub': 0.2,
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

        Lq = (self.parameters.Wq / 2.) * np.sum(q_err**2, axis=0)
        Lw = (self.parameters.Ww / 2.) * np.sum(w**2, axis=0)
        Lu = (self.parameters.Wu / 2.) * np.sum(self._saturate(u)**2, axis=0)

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
            dLdu = self.parameters.Wu * self._saturate(u)

            # Where the control is saturated, the gradient is zero
            sat_idx = self._find_saturated(u)
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

            # Where the control is saturated, the gradient is zero (constant).
            # This makes the Hessian zero in all terms that include a saturated
            # control
            sat_idx = self._find_saturated(u)
            sat_idx = sat_idx[None, ...] + sat_idx[:, None, ...]
            R[sat_idx] = 0.

            if squeeze:
                R = R[..., 0]
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        q, q0, w = self._break_state(x)
        u = self._saturate(u)

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

            # Where the control is saturated, the Jacobian is zero
            sat_idx = self._find_saturated(u)
            dfdu[:, sat_idx] = 0.

            if squeeze:
                dfdu = dfdu[..., 0]
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        u = np.matmul(self.parameters._Jinv.T, p[4:]) / self.parameters.Wu
        return self._saturate(u)

    def optimal_control_jac(self, x, p, u0=None):
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

        u = self.optimal_control(x, p)
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

    def _constraint_fun(self, X):
        """
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
        """
        return 1. - np.sum(X[:4]**2, axis=0, keepdims=True)

    def _constraint_jacobian(self, X):
        """
        Constraint function Jacobian dC/dX of self.constraint_fun.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        JC : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        """
        JC = -2. * X[:4]
        return np.hstack((JC.reshape(1,-1), np.zeros((1,3))))
