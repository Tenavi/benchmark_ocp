"""
The `controls` module contains the `Controller` template class for implementing
feedback control policies. It is recommended that any user-defined feedback
controllers be implemented as subclasses of `Controller`. The
`LinearQuadraticRegulator` is implemented as a simple example of this usage.
"""

import pickle

import numpy as np
from scipy.linalg import solve_continuous_are
from sklearn.metrics import r2_score

from . import utilities


class Controller:
    """Base class for implementing a state feedback controller."""
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        return type(self).__name__

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

    def r2_score(self, x_data, u_data, multioutput='uniform_average'):
        r"""
        Return the coefficient of determination of the control prediction in the
        physical (unscaled) domain.

        The coefficient of determination, $R^2$, is defined as
        `r2 = 1 - residual / total`, where
        `residual = ((u_data - u_pred)**2).sum()` with `u_pred = self(x_data)`,
        and `total = ((u_data - u_data.mean()) ** 2).sum()`. The best possible
        score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of `u_data`, disregarding the input features, would get an $R^2$
        score of 0.0.

        Parameters
        ----------
        x_data : (n_states, n_data) array
            A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        u_data : (n_controls, n_data) array
            The optimal feedback controls evaluated at the states `x_data`.
        multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
                (n_controls,) array, or None, default='uniform_average'

            Defines aggregating of multiple output scores. An array value
            defines weights used to average scores, and None reverts to the
            default 'uniform_average'.

                * 'raw_values' : Returns a full set of scores for each control.
                * 'uniform_average' :
                    Scores of all control dimensions are averaged with uniform
                    weight.
                * 'variance_weighted' :
                    Scores of all control dimensions are averaged, weighted by
                    the variances of each individual control.

        Returns
        -------
        r2 : float or (n_controls,) array
            The $R^2$ score, or array of scores if `multioutput=='raw_values'`.
        """
        u_pred = self(x_data)
        u_data = np.reshape(u_data, u_pred.shape)
        return r2_score(u_data.T, u_pred.T, multioutput=multioutput)

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
            if np.ndim(B) > 2:
                B = np.squeeze(B)
            if np.ndim(R) > 2:
                R = np.squeeze(R)

            if not hasattr(self, 'P'):
                self.P = self.solve_care(A, B, Q, R)

            self._RB = np.linalg.solve(R, np.transpose(B))
            self.K = np.matmul(self._RB, self.P)

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
        algebraic Riccati equations (CARE), where one or more columns of `A` and
        `Q` can be all zeros, which can happen for certain linearized systems.
        In such situations, these zero-column states don't impact dynamics of
        other states or the cost function. The CARE solver will often fail for
        the full set of states, but can find a solution to the sub-problem which
        ignores the zero-column states. Using the resulting control law in the
        full system stabilizes the controlled states, with the ignored states'
        stability depending on their dynamics.

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
            Absolute tolerance when comparing elements of `A` and `Q` to zero.

        Returns
        -------
        P : (n_states, n_states) array
            Solution to the continuous-time algebraic Riccati equation. If any
            columns of `A` and `Q` are all zeros, these columns of `P` will also
            be zero.

        Raises
        ------
        LinAlgError
            For cases where the stable subspace of the pencil could not be
            isolated. See `scipy.linalg.solve_continuous_are` for details.
        """
        n = np.shape(A)[0]

        A = np.reshape(A, (n, n))
        Q = np.reshape(Q, (n, n))
        B = np.reshape(B, (n, -1))
        R = np.reshape(R, (B.shape[1], B.shape[1]))

        A_zero_idx = np.isclose(A, 0., atol=zero_tol).all(axis=0)
        Q_zero_idx = np.isclose(Q, 0., atol=zero_tol).all(axis=0)
        non_zero_idx = ~ np.all([A_zero_idx, Q_zero_idx], axis=0)

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
            return u[:, 0]

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

        u0 = np.reshape(u0, (self.n_controls, -1))
        zero_idx = utilities.find_saturated(u0, lb=self.u_lb, ub=self.u_ub)

        if np.ndim(x) < 2:
            dudx = - self.K
            zero_idx = np.squeeze(zero_idx, axis=-1)
        else:
            dudx = np.tile(- self.K[:, None], (1, np.shape(x)[1], 1))
            dudx = np.moveaxis(dudx, 1, 2)
            zero_idx = np.tile(zero_idx[:, None, :], (1, np.shape(x)[0], 1))

        dudx[zero_idx] = 0.

        return dudx


class ConstantControl(Controller):
    """A `Controller` subclass which returns a single constant value for all
    states, used for some unit tests and for simulating uncontrolled systems."""
    def __init__(self, u):
        """
        Parameters
        ----------
        u : (n_controls, 1) array
            The constant value to return for all state inputs.
        """
        self.n_controls = np.size(u)
        self.u = np.reshape(u, (self.n_controls, 1))

    def __call__(self, x):
        if np.ndim(x) < 2:
            return self.u[:, 0]

        return np.tile(self.u, (1, np.shape(x)[1]))

    def jac(self, x, u0=None):
        return np.zeros((self.n_controls,) + np.shape(x))


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
