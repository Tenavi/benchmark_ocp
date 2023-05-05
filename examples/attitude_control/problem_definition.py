import numpy as np

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import saturate, resize_vector
from optimalcontrol.sampling import UniformSampler


def cross_product_matrix(w):
    zeros = np.zeros_like(w[0])
    wx = np.array([[zeros, -w[2], w[1]],
                   [w[2], zeros, -w[0]],
                   [-w[1], w[0], zeros]])
    return wx


class AttitudeControl(OptimalControlProblem):
    _required_parameters = {'J': [[59.22, -1.14, -0.8],
                                  [-1.14, 40.56, 0.1],
                                  [-0.8, 0.1, 57.6]],
                            'Wq': 1., 'Ww': 1., 'Wu': 1.,
                            'x0_max_rate_deg': 5.}
    _optional_parameters = {'u_lb': -0.3, 'u_ub': 0.3, 'x0_sample_seed': None,
                            'x0_sample_norm': 2}

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
        if 'J' in new_params:
            try:
                obj.J = np.reshape(obj.J, (3, 3))
            except:
                raise ValueError('Inertia matrix J must be (3, 3)')
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

        w0_ub = np.deg2rad(np.abs(obj.x0_max_rate_deg))
        w0_ub = resize_vector(w0_ub, 3)

        if not hasattr(self, '_x0_sampler'):
            self._w0_sampler = UniformSampler(
                lb=-w0_ub, ub=w0_ub, norm=getattr(obj, 'x0_sample_norm', 2),
                seed=getattr(obj, 'x0_sample_seed', None))
        elif 'x0_max_rate_deg' in new_params or 'x0_sample_seed' in new_params:
            self._x0_sampler.update(lb=-w0_ub, ub=w0_ub,
                                    seed=new_params.get('x0_sample_seed'))

    def _break_state(self, X):
        '''
        Break up the state vector, [q0, q, w], into individual pieced. If
        X.shape[0] >= 14, then splits off the remaining states (costates) too.

        Parameters
        ----------
        X : (7,), (14,), (7, n_samples), or (14, n_samples) array
            State (and costate, if X.shape[0] > 7)

        Returns
        -------
        q0 : (1,) or (1, n_samples) array
            Scalar component of quaternion
        q : (4,) or (4, n_samples) array
            Vector component of quaternion
        w : (3,) or (3, n_samples) array
            Angular momenta
        A0 : (1,) or (1, n_samples) array
            Costate of scalar component of quaternion
        Aq : (4,) or (4, n_samples) array
            Costate of vector component of quaternion
        Aw : (3,) or (3, n_samples) array
            Costate of angular momenta
        '''
        q0 = X[:1]
        q = X[1:4]
        w = X[4:7]

        if X.shape[0] > 7:
            A0 = X[7:8]
            Aq = X[8:11]
            Aw = X[11:14]

            return q0, q, w, A0, Aq, Aw

        return q0, q, w

    def sample_X0(self, Ns, dist=None):
        # Samples angular velocities
        w = np.random.rand(3, Ns)
        w = (self.X0_ub[4:] - self.X0_lb[4:]) * w + self.X0_lb[4:]

        # Samples Euler angles
        v = (2.*np.pi) * np.random.rand(3, Ns) - np.pi
        v[1] *= 0.5

        # Converts to quaternions
        cos_v = np.cos(v / 2.)
        sin_v = np.sin(v / 2.)

        q = np.vstack((
            -cos_v[0]*sin_v[1]*sin_v[2] + cos_v[1]*cos_v[2]*sin_v[0],
            cos_v[0]*cos_v[2]*sin_v[1] + sin_v[0]*cos_v[1]*sin_v[2],
            cos_v[0]*cos_v[1]*sin_v[2] - sin_v[0]*cos_v[2]*sin_v[1]
        ))

        if dist is not None:
            if dist <= 1.:
                w = np.zeros_like(w)
            else:
                w_dist = np.sqrt(dist**2 - 1)
                dist = 1
                w *= w_dist / np.linalg.norm(w, axis=0)

            q *= dist / np.linalg.norm(q, axis=0)

        # Sets scalar quaternion positive
        q0 = np.sqrt(1. - np.sum(q**2, axis=0, keepdims=True))
        X0 = np.vstack((q0, q, w))

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def running_cost(self, X, U):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        _, q, w = self._break_state(X[:7])

        return (
            self.Wq/2. * np.sum(q**2, axis=0)
            + self.Ww/2. * np.sum(w**2, axis=0)
            + self.Wu/2. * np.sum(U**2, axis=0)
        )

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdX : bool, default=True
            Set to True to compute the gradient with respect to states, dL/dX.
        return_dLdU : bool, default=True
            Set to True to compute the gradient with respect to controls, dL/dU.

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        q0, q, w = self._break_state(X[:7])

        if return_dLdX:
            dLdX = np.concatenate((np.zeros_like(q0), self.Wq * q, self.Ww * w))
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = self.Wu * U
            if not return_dLdX:
                return dLdU

        return dLdX, dLdU

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