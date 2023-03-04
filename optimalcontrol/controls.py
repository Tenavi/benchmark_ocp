import numpy as np

from . import utilities as utils

class Controller:
    """Base class for implementing a state feedback controller."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        """
        Evaluates the feedback control, u(x), for each sample state in x.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        u : (n_controls, n_points) or (n_controls,) array
            Feedback control for each column in x.
        """
        raise NotImplementedError

    def jacobian(self, x, u0=None):
        """
        Evaluates the Jacobian of the feedback control, [du/dx](x), for each
        sample state in x. Default implementation uses finite differences.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.
        u0 : (n_states, n_controls) or (n_controls,) array, optional
            `self(x)`, pre-evaluated at the inputs.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of feedback control for each column in x.
        """
        return utils.approx_derivative(self, f0=u0)

class LinearQuadraticRegulator(Controller):
    """Linear quadratic regulator (LQR) control with saturation constraints."""
    def __init__(
            self, A=None, B=None, Q=None, R=None,
            u_lb=None, u_ub=None, xf=0., uf=0., P=None, K=None
        ):
        """
        Parameters
        ----------
        A : (n_states, n_states) array, optional
            State Jacobian matrix. Required if `K` is `None`.
        B : (n_states, n_controls) array, optional
            Control Jacobian matrix. Required if `K` is `None`.
        Q : (n_states, n_states) array, optional
            Hessian of running cost with respect to states. Must be positive
            semi-definite. Required if `K` is `None`.
        R : (n_controls, n_controls) array, optional
            Hessian of running cost with respect to controls. Must be positive
            definite. Required if `K` is `None`.
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            upper control saturation bounds.
        xf : (n_states, 1) array, default=0.
            Goal state, nominal linearization point.
        uf : (n_controls, 1) array, default=0.
            Control values at nominal linearization point.
        P : (n_states, n_states) array, optional
            Previously-computed Riccati equation solution for this problem.
        K : (n_controls, n_states) array, optional
            Previously-computed control gain matrix for this problem.
            `K = inv(R) @ B.T @ P` and `u(x) = uf - K @ (x - xf)`.
        """
        if P is not None:
            self.P = np.asarray(P)

        # Compute LQR control gain matrix
        if K is not None:
            self.K = np.asarray(K)
        else:
            from scipy.linalg import solve_continuous_are
            self.P = solve_continuous_are(A, B, Q, R)

            self.RB = np.linalg.solve(R, np.transpose(B))
            self.K = np.matmul(self.RB, self.P)

        self.xf = utils.resize_vector(xf, self.K.shape[1])
        self.uf = utils.resize_vector(uf, self.K.shape[0])

        self.n_states = self.xf.shape[0]
        self.n_controls = self.uf.shape[0]

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = utils.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = utils.resize_vector(self.u_ub, self.n_controls)

    def __call__(self, x):
        x_err = np.reshape(x, (self.n_states, -1)) - self.xf
        u = self.uf - np.matmul(self.K, x_err)
        u = utils.saturate(u, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u.flatten()

        return u

    def jacobian(self, x, u0=None):
        u = self(x)

        if np.ndim(x) < 2:
            dudx = - self.K
        else:
            dudx = np.tile(- self.K[:,None], (1,np.shape(x)[1],1))
            dudx = np.moveaxis(dudx, 1, 2)

        zero_idx = utils.find_saturated(u, min=self.u_lb, max=self.u_ub)

        dudx[zero_idx] = 0.
        return dudx
