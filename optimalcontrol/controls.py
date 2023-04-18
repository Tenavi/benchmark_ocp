"""
The `controls` module contains the `Controller` template class for implementing
feedback control policies. It is recommended that any user-defined feedback
controllers be implemented as subclasses of `Controller`. The
`LinearQuadraticRegulator` is implemented as a simple example of this usage.
"""

import numpy as np

from . import utilities


class Controller:
    """Base class for implementing a state feedback controller."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        """
        Evaluates the feedback control, `u(x)`, for each sample state in `x`.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        u : (n_controls, n_points) or (n_controls,) array
            Feedback control for each column in `x`.
        """
        raise NotImplementedError

    def jac(self, x, u0=None):
        """
        Evaluates the Jacobian of the feedback control, $du/dx (x)$, for each
        sample state in `x`. Default implementation uses finite differences.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.
        u0 : (n_states, n_controls) or (n_controls,) array, optional
            `self(x)`, pre-evaluated at the inputs.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of feedback control for each column in `x`.
        """
        return utilities.approx_derivative(self, x, f0=u0)


class LinearQuadraticRegulator(Controller):
    """Linear quadratic regulator (LQR) control with saturation constraints."""
    def __init__(self, A=None, B=None, Q=None, R=None, K=None, P=None,
                 u_lb=None, u_ub=None, xf=0., uf=0.):
        """
        Parameters
        ----------
        A : (n_states, n_states) array, optional
            State Jacobian matrix, $df/dx (x_f, u_f)$. Required if `K` is
            `None`.
        B : (n_states, n_controls) array, optional
            Control Jacobian matrix, $df/du (x_f, u_f)$. Required if `K` is
            `None`.
        Q : (n_states, n_states) array, optional
            Hessian of running cost with respect to states,
            $d^2L/dx^2 (x_f, u_f)$. Must be positive semi-definite. Required if
            `K` is `None`.
        R : (n_controls, n_controls) array, optional
            Hessian of running cost with respect to controls,
            $d^2L/du^2 (x_f, u_f)$. Must be positive definite. Required if `K`
            is `None`.
        K : (n_controls, n_states) array, optional
            Previously-computed control gain matrix for this problem. The LQR
            control law is `u(x) = sat(uf - K @ (x - xf))`, where
            `K = inv(R) @ B.T @ P` and `sat` is the saturation function.
        P : (n_states, n_states) array, optional
            Previously-computed solution to the continuous algebraic Riccati
            equation.
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            Upper control saturation bounds.
        xf : {(n_states, 1) array, float}, default=0.
            Goal state, nominal linearization point. If float, will be broadcast
            into an array of shape `(n_states, 1)`.
        uf : {(n_controls, 1) array, float}, default=0.
            Control values at nominal linearization point. If float, will be
            broadcast into an array of shape `(n_controls, 1)`.
        """
        if P is not None:
            self.P = np.asarray(P)

        # Compute LQR control gain matrix
        if K is not None:
            self.K = np.asarray(K)
        else:
            if not hasattr(self, 'P'):
                from scipy.linalg import solve_continuous_are
                self.P = solve_continuous_are(A, B, Q, R)
            self.RB = np.linalg.solve(R, np.transpose(B))
            self.K = np.matmul(self.RB, self.P)

        self.xf = utilities.resize_vector(xf, self.K.shape[1])
        self.uf = utilities.resize_vector(uf, self.K.shape[0])

        self.n_states = self.xf.shape[0]
        self.n_controls = self.uf.shape[0]

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = utilities.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = utilities.resize_vector(self.u_ub, self.n_controls)

    def __call__(self, x):
        x_err = np.reshape(x, (self.n_states, -1)) - self.xf
        u = self.uf - np.matmul(self.K, x_err)
        u = utilities.saturate(u, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u.flatten()

        return u

    def jac(self, x, u0=None):
        """
        Evaluates the Jacobian of the feedback control, $du/dx (x)$, for each
        sample state in `x`. For LQR this is just the feedback gain matrix,
        `-self.K`, copied to the appropriate size. When the control is
        saturated, infinitesimal changes in the state will not change the
        control so the Jacobian is zero for states that saturate the control.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.
        u0 : (n_states, n_controls) or (n_controls,) array, optional
            `self(x)`, pre-evaluated at the inputs.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of feedback control for each column in `x`.
        """
        if u0 is None:
            u0 = self(x)

        if np.ndim(x) < 2:
            dudx = - self.K
        else:
            dudx = np.tile(- self.K[:, None], (1, np.shape(x)[1], 1))
            dudx = np.moveaxis(dudx, 1, 2)

        zero_idx = utilities.find_saturated(u0, lb=self.u_lb, ub=self.u_ub)

        dudx[zero_idx] = 0.
        return dudx
