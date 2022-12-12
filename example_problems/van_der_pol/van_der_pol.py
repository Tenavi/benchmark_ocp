import numpy as np

from ..example_config import Config
from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.utilities import saturate, UniformSampler

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
    _params = {
        'Wx': .5, 'Wy': 1., 'Wu': 4., 'xf': 0.,
        'mu': 2., 'b': 1.5, 'u_max': 1.,
        'x0_ub': np.array([[3.],[4.]]), 'x0_lb': -np.array([[3.],[4.]])
    }
    def _saturate(self, u):
        return saturate(u, -self.u_max, self.u_max)

    @property
    def n_states(self):
        '''The number of system states (positive int).'''
        return 2

    @property
    def n_controls(self):
        '''The number of control inputs to the system (positive int).'''
        return 1

    @property
    def final_time(self):
        '''Time horizon of the system.'''
        return np.inf

    def _update_params(self, **new_params):
        if 'xf' in new_params or not hasattr(self, 'xf'):
            self.xf = np.zeros((2,1))
            self.xf[0] = self._params.xf

        if 'b' in new_params or not hasattr(self, 'B'):
            self.B = np.zeros((2,1))
            self.B[1] = self._params.b

        if 'b' in new_params or 'xf' in new_params or not hasattr(self, 'uf'):
            self.uf = self.xf[0] / self._params.b

        if 'u_max' in new_params or not hasattr(self, 'u_max'):
            self.u_max = np.abs(self._params.u_max)

        if not hasattr(self, '_x0_sampler'):
            self._x0_sampler = UniformSampler(
                lb=self._params.x0_lb, ub=self._params.x0_ub, xf=self.xf,
                norm=getattr(self._params, 'x0_sample_norm', 1),
                seed=getattr(self._params, 'x0_sample_seed', None)
            )
        elif any([
                'lb' in new_params,
                'ub' in new_params,
                'x0_sample_seed' in new_params,
                'xf' in new_params
            ]):
            self._x0_sampler.update(
                lb=new_params.get('lb'),
                ub=new_params.get('ub'),
                xf=self.xf,
                seed=new_params.get('x0_sample_seed')
            )

    def sample_initial_conditions(self, n_samples=1, distance=None):
        '''
        Generate initial conditions uniformly from a hypercube.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance (in l1 or l2 norm) of samples from `self.xf`. The
            type of norm is determined by `self.norm`. Note that depending on
            how `distance` is specified, samples may be outside the hypercube.

        Returns
        -------
        x0 : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x0` will be a one-dimensional array.
        '''
        return self._x0_sampler(n_samples=n_samples, distance=distance)

    def running_cost(self, x, u):
        '''
        Evaluate the running cost L(x,u) at one or more state-control pairs.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : float or (n_points,) array
            Running cost(s) L(x,u) evaluated at pair(s) (x,u).
        '''
        if x.ndim < 2:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf

        u_err = self._saturate(u) - self.uf

        x_err = x_err**2

        L = (self._params.Wx/2.) * x_err[:1]
        L += (self._params.Wy/2.) * x_err[1:]
        L += (self._params.Wu/2.) * u_err**2

        return L[0]

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
        dLdu : (controls,) or (n_controls, n_points) array
            Control gradients dL/du (x,u) evaluated at pair(s) (x,u).
        '''
        if x.ndim < 2:
            x_err = x - self.xf.flatten()
        else:
            x_err = x - self.xf

        u_err = self._saturate(u) - self.uf

        x1 = x_err[:1]
        x2 = x_err[1:]

        if return_dLdx:
            dLdx = np.concatenate((self._params.Wx * x1, self._params.Wy * x2))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self._params.Wu * u_err
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hessians(self, x, u, return_dLdx=True, return_dLdu=True):
        '''
        Evaluate the Hessians of the running cost, d^2L/dx^2 (x,u) and
        d^2L/du^2 (x,u), at one or multiple state-control pairs. Default
        implementation approximates this with central differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdx : bool, default=True
            If True, compute the Hessian with respect to states, dL/dx.
        return_dLdu : bool, default=True
            If True,compute the Hessian with respect to controls, dL/du.

        Returns
        -------
        dLdx : (n_states, n_states) or (n_states, n_states, n_points) array
            State Hessians dL^2/dx^2 (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_controls,) or (n_controls, n_controls, n_points) array
            Control Hessians dL^2/du^2 (x,u) evaluated at pair(s) (x,u).
        '''
        if return_dLdx:
            Q = np.diag([self._params.Wx, self._params.Wy])
            if x.ndim >= 2:
                Q = np.tile(Q[...,None], (1,1,x.shape[1]))
            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.reshape(self._params.Wu, (1,1))
            if u.ndim >=2:
                R = np.tile(R[...,None], (1,1,u.shape[1]))
            if not return_dLdx:
                return R

        return Q, R

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
        u = self._saturate(u)
        if x.ndim < 2 and u.ndim > 1:
            u = u.flatten()

        x1 = x[:1]
        x2 = x[1:]

        dx1dt = x2
        dx2dt = self._params.mu * (1. - x1**2) * x2 - x1 + self._params.b * u

        return np.concatenate((dx1dt, dx2dt), axis=0)

    def jacobians(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances. Default implementation
        approximates the Jacobians with central differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dfdx : bool, default=True
            If True, compute the Jacobian with respect to states, df/dx.
        return_dfdu : bool, default=True
            If True, compute the Jacobian with respect to controls, df/du.
        f0 : ignored
            For API consistency only.

        Returns
        -------
        dfdx : (n_states, n_states) or (n_states, n_states, n_points) array
            Jacobian with respect to states.
        dfdu : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Jacobian with respect to controls.
        '''
        if return_dfdx:
            dfdx = np.zeros((self.n_states,) + x.shape)
            dfdx[0,1] = 1.
            dfdx[1,0] = -1. - 2.*self._params.mu*x[0]*x[1]
            dfdx[1,1] = self._params.mu*(1. - x[0]**2)

            if not return_dfdu:
                return dfdx

        if return_dfdu:
            if x.ndim > 1:
                dfdu = np.tile(self.B[...,None], (1,1,x.shape[-1]))
            else:
                dfdu = np.copy(self.B)

            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        u : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        u = self.uf - self._params.b / self._params.Wu * p[1:]
        return self._saturate(u)

    def optimal_control_jacobian(self, x, p, u0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        u0 : ignored
            For API consistency only.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, du/dx (x; p).
        '''
        if p.ndim < 2:
            return np.zeros((self.n_controls, self.n_states))
        return np.zeros((self.n_controls, self.n_states, p.shape[-1]))

    def bvp_dynamics(self, t, xp):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        t : (n_points,) array
            Time collocation points for each state.
        xp : (2*n_states, n_points) array
            Current state and costate.

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
        dx2dt = self._params.mu * (1. - x1**2) * x2 - x1 + self._params.b * u

        # Costate dynamics
        dp1dt = -self._params.Wx * x1_err + p2 * (2.*self._params.mu*x1*x2 + 1.)
        dp2dt = -self._params.Wy * x2 - p1 - p2 * self._params.mu * (1. - x1**2)

        return np.vstack((dx1dt, dx2dt, dp1dt, dp2dt))
