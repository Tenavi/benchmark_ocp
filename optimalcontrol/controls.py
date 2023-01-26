import numpy as np
from scipy.optimize._numdiff import approx_derivative

from .utilities import saturate

class Controller:
    """
    Base class for implementing a state feedback controller.
    """
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

    def jacobian(self, x):
        """
        Evaluates the Jacobian of the feedback control, [du/dx](x), for each
        sample state in x.

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of feedback control for each column in x.
        """
        raise NotImplementedError

class LinearQuadraticRegulator(Controller):
    """
    Implements a linear quadratic regulator (LQR) control with saturation
    constraints.
    """
    def __init__(
            self, xf, uf, A=None, B=None, Q=None, R=None,
            u_lb=None, u_ub=None, P=None
        ):
        """
        Parameters
        ----------
        xf : (n_states, 1) array
            Goal state, nominal linearization point.
        uf : (n_controls, 1) array
            Control values at nominal linearization point.
        A : (n_states, n_states) array, optional
            State Jacobian matrix at nominal equilibrium. Required if
            `P is None`.
        B : (n_states, n_controls) array, optional
            Control Jacobian matrix at nominal equilibrium. Required if
            `P is None`.
        Q : (n_states, n_states) array, optional
            Hessian of running cost with respect to states. Must be positive
            semi-definite. Required if `P is None`.
        R : (n_controls, n_controls) array, optional
            Hessian of running cost with respect to controls. Must be positive
            definite. Required if `P is None`.
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            upper control saturation bounds.
        P : (n_states, n_states) array, optional
            Previously-computed Riccati equation solution for this problem.
        """
        self.xf = np.reshape(xf, (-1,1))
        self.uf = np.reshape(uf, (-1,1))

        self.n_states = self.xf.shape[0]
        self.n_controls = self.uf.shape[0]

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = np.reshape(self.u_lb, (self.n_controls,1))
        if self.u_ub is not None:
            self.u_ub = np.reshape(self.u_ub, (self.n_controls,1))

        # Make Riccati matrix and LQR control gain matrix
        if P is not None:
            self.P = np.asarray(P)
        else:
            from scipy.linalg import solve_continuous_are
            self.P = solve_continuous_are(A, B, Q, R)

        self.RB = np.linalg.solve(R, np.transpose(B))
        self.K = np.matmul(self.RB, self.P)

    def __call__(self, x):
        x_err = np.reshape(x, (self.n_states, -1)) - self.xf
        u = self.uf - np.matmul(self.K, x_err)
        u = saturate(u, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u.flatten()

        return u

    def jacobian(self, x):
        u = self(x)

        if np.ndim(x) < 2:
            dudx = - self.K
        else:
            dudx = np.tile(- self.K[:,None], (1,np.shape(x)[1],1))
            dudx = np.moveaxis(dudx, 1, 2)

        if self.u_ub is not None and self.u_lb is not None:
            zero_idx = np.any([self.u_ub <= u, u <= self.u_lb], axis=0)
        elif self.u_ub is not None:
            zero_idx = self.u_ub <= u
        elif self.u_lb is not None:
            zero_idx = u <= self.u_ub
        else:
            return dudx

        dudx[zero_idx] = 0.
        return dudx
