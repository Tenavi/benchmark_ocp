import numpy as np

from ..example_config import Config
from ocp.problems import OptimalControlProblem, LinearProblem
from ocp.utilities import saturate

config = Config(
    ode_solver='RK23',
    atol=1e-08,
    rtol=1e-04,
    t1_sim=20.,
    t1_max=120.,
    n_trajectories_train=50,
    n_trajectories_test=50
)

class VanDerPol(OptimalControlProblem):
    def __init__(self):
        # Dynamics parameters
        self.mu = 2.
        self.b = 1.5

        # Cost parameters
        self.Wx = 1/2
        self.Wy = 1.
        self.Wu = 4.

        # Control constraints
        u_max = 1.
        if u_max is None:
            self.u_lb, self.u_ub = None, None
        else:
            self.u_lb = np.full((1, 1), -u_max)
            self.u_ub = np.full((1, 1), u_max)

        # Initial condition bounds
        X0_ub = np.array([[3.],[4.]])
        X0_lb = - X0_ub

        # Linearization point
        xf = [0., 0.]
        uf = xf[0] / self.b

        # Dynamics linearized around xf (dxdt ~= Ax + Bu)
        A = [[0., 1.], [-1., self.mu*(1. - xf[0]**2)]]
        self.B = np.array([[0.], [self.b]])

        # Cost matrices
        Q = np.diag([self.Wx / 2., self.Wy / 2.])
        R = [[self.Wu / 2.]]

        super().__init__(n_states=2, n_controls=1, u_lb=u_max, u_ub=u_max)

        self.linearizations.append(LinearProblem(xf, uf, A, self.B, Q, R))

        self.xf = self.linearizations[0].xf
        self.uf = self.linearizations[0].uf

    def running_cost(self, x, u):
        '''
        Evaluate the running cost L(x,u) at one or multiple state-control pairs.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(x,u) evaluated at pair(s) (x,u).
        '''
        if x.ndim < 1:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf

        if u.ndim < 1:
            u_err = u - self.uf.flatten()
        else:
            u_err = u - self.uf

        x1 = x_err[:1]
        x2 = x_err[1:]

        L = (self.Wx/2.)*x1**2 + (self.Wy/2.)*x2**2 + (self.Wu/2.)*u_err**2
        return np.squeeze(L)

    def running_cost_gradients(self, x, u, return_dLdx=True, return_dLdu=True):
        '''
        Evaluate the gradients of the running cost, dL/dx (x,u) and dL/du (x,u),
        at one or multiple state-control pairs. Default implementation
        approximates this with central differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdx : bool, default=True
            If True, compute the gradient with respect to states, dL/dx.
        return_dLdu : bool, default=True
            If True,compute the gradient with respect to controls, dL/du.

        Returns
        -------
        dLdx : (n_states,) or (n_states, n_points) array
            State gradients dL/dx (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_states,) or (n_states, n_points) array
            Control gradients dL/du (x,u) evaluated at pair(s) (x,u).
        '''
        if x.ndim < 1:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf

        if u.ndim < 1:
            u_err = u - self.uf.flatten()
        else:
            u_err = u - self.uf

        x1 = x_err[:1]
        x2 = x_err[1:]

        if return_dLdX:
            dLdx = np.concatenate((self.Wx * x1, self.Wy * x2))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self.Wu * u_err
            if not return_dLdx:
                return dLdu

        return dLdu, dLdu

    def dynamics(self, x, u):
        '''
        Evaluate the closed-loop dynamics at single or multiple time instances.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dxdt : (n_states,) or (n_states, n_points) array
            System dynamics dx/dt = f(x,u).
        '''
        if x.ndim < 2:
            u = u.flatten()

        x1 = x[:1]
        x2 = x[1:]

        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * u

        return np.concatenate((dx1dt, dx2dt))

    def jacobians(self, x, u, f0=None):
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        f0 : ignored
            For API consistency only.

        Returns
        -------
        dfdx : (n_states, n_states, n_points) array
            Jacobian with respect to states.
        dfdu : (n_states, n_controls, n_points) array
            Jacobian with respect to controls.
        '''
        x1 = np.atleast_1d(x[0])
        x2 = np.atleast_1d(x[1])

        dfdx = np.array([
            [np.zeros_like(x1), np.ones_like(x1)],
            [-1. - 2.*self.mu*x1*x2, self.mu*(1. - x1**2)]
        ])
        dfdu = np.expand_dims(self.B, -1)
        dfdu = np.tile(dfdu, (1,1,x1.shape[-1]))

        return dfdu, dfdu

    def optimal_control(self, x, dVdx):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdx : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        u : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        u = self.uf - self.b / self.Wu * dVdx[1:]
        return saturate(u, self.u_lb, self.u_ub)

    def optimal_control_jac(self, x, dVdx, u0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdx : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        u0 : ignored
            For API consistency only.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, du/dx (x; dVdx).
        '''
        if dVdx.ndim < 2:
            return np.zeros((self.n_controls, self.n_states))
        return np.zeros((self.n_controls, self.n_states, dVdx.shape[-1]))

    def bvp_dynamics(self, t, xp):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        t : (n_points,) array
            Time collocation points for each state.
        xp : (2*n_states, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dxpdt : (2*n_states, n_points) array
            Concatenation of dynamics dx/dt = f(x,u^*) and costate dynamics,
            dp/dt = -dH/dx(x,u^*,p), where u^* is the optimal control.
        '''
        u = self.optimal_control(xp[:2], xp[2:])

        x1 = xp[:1]
        x2 = xp[1:2]

        x1_err = x1 - self.xf[:1]

        # Costate
        p1 = xp[2:3]
        p2 = xp[3:4]

        # State dynamics
        dx1dt = x2
        dx2dt = self.mu * (1. - x1**2) * x2 - x1 + self.b * u

        # Costate dynamics
        dp1dt = -self.Wx * x1_err + p2 * (2.*self.mu*x1*x2 + 1.)
        dp2dt = -self.Wy * x2 - p1 - p2 * self.mu * (1. - x1**2)

        return np.vstack((dx1dt, dx2dt, dp1dt, dp2dt))
