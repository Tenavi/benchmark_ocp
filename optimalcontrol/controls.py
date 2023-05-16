"""
The `controls` module contains the `Controller` template class for implementing
feedback control policies. It is recommended that any user-defined feedback
controllers be implemented as subclasses of `Controller`. The
`LinearQuadraticRegulator` is implemented as a simple example of this usage.
"""

import pickle

import numpy as np
from scipy.linalg import solve_continuous_are

from . import utilities


def from_pickle(filepath):
    """
    WARNING: `pickle` is not secure. Only unpickle files you trust.

    Load a `Controller` object from a pickle file.

    Parameters
    ----------
    filepath : path_like
        Path to where a `Controller` object is saved.

    Returns
    -------
    controller : `Controller`
        Unpickled `Controller` instance.
    """
    with open(filepath, 'rb') as file:
        controller = pickle.load(file)
    return controller


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

    def pickle(self, filepath):
        """
        Save the `Controller` object using `pickle`.

        Parameters
        ----------
        filepath : path_like
            Path to a file where the `Controller` should be saved.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)


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
            A = np.squeeze(A)
            B = np.squeeze(B)
            Q = np.squeeze(Q)
            R = np.squeeze(R)

            if not hasattr(self, 'P'):
                self.P = self.solve_care(A, B, Q, R)
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

    @staticmethod
    def solve_care(A, B, Q, R, zero_tol=1e-12):
        r"""
        Wrapper of `scipy.linalg.solve_continuous_are` to solve continuous-time
        algebraic Riccati equations (CARE) where one or more rows of `A` and `Q`
        are all zeros, which can happen for certain linearized systems. For
        details see `scipy.linalg.solve_continuous_are`.

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
        zero_tol : float, default=1e-12
            Absolute tolerance when comparing elements of `A`, `B` and `Q` to
            zero.

        Returns
        -------
        P : (n_states, n_states) array
            Solution to the continuous-time algebraic Riccati equation.

        Raises
        ------
        LinAlgError
            For cases where the stable subspace of the pencil could not be
            isolated. See Notes section and the references for details.
        """
        n = np.shape(A)[0]

        A = np.reshape(A, (n, n))
        Q = np.reshape(Q, (n, n))
        B = np.reshape(B, (n, -1))
        R = np.reshape(R, (B.shape[1], B.shape[1]))

        A_zero_idx = np.isclose(A, 0., atol=zero_tol).all(axis=-1)
        Q_zero_idx = np.isclose(Q, 0., atol=zero_tol).all(axis=-1)
        B_zero_idx = np.isclose(B, 0., atol=zero_tol).all(axis=-1)
        non_zero_idx = ~ np.all([A_zero_idx, Q_zero_idx, B_zero_idx], axis=0)

        A = A[non_zero_idx][:, non_zero_idx]
        Q = Q[non_zero_idx][:, non_zero_idx]
        B = B[non_zero_idx]

        P = np.zeros((n, n))
        P_sub = np.zeros((A.shape[0], n))
        P_sub[:, non_zero_idx] = solve_continuous_are(A, B, Q, R)
        P[non_zero_idx] = P_sub

        return P

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
