import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.spatial.distance import cdist

from .utilities import approx_derivative, saturate

class ProblemParameters:
    '''Utility class to store cost function and system dynamics parameters.'''
    def __init__(self, **params):
        self.__dict__.update(params)
        self._update_fun = None

    def update(self, **params):
        '''Modify individual or multiple parameters using keyword arguments.'''
        self.__dict__.update(params)
        self.update_fun(**params)

    @property
    def update_fun(self):
        '''Get or set a function to execute whenever problem parameters are
        modified by `update`.'''
        if not callable(self._update_fun):
            raise ValueError("update_fun has not been set.")
        return self._update_fun

    @update_fun.setter
    def update_fun(self, _update_fun):
        if not callable(_update_fun):
            raise TypeError("update_fun must be set with a callable.")
        self._update_fun = _update_fun

class OptimalControlProblem:
    '''
    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.
    '''
    # Default cost and system parameters. Should be overwritten by subclass.
    _params = ProblemParameters()
    # Finite difference method for default gradient, Jacobian, and Hessian
    # approximations
    _fin_diff_method = '3-point'

    def __init__(self, **problem_parameters):
        '''
        Initialize the OCP with some cost and system parameters.
        Parameters
        ----------
        problem_parameters : dict, default={}
            Parameters specifying the cost function and system dynamics. If
            empty, defaults defined by the subclass will be used.
        '''
        if not isinstance(self._params, ProblemParameters):
            self._params = ProblemParameters(**self._params)
        self.parameters.update_fun = self._update_params
        self.parameters.update(**problem_parameters)

    @property
    def n_states(self):
        '''The number of system states (positive int).'''
        raise NotImplementedError

    @property
    def n_controls(self):
        '''The number of control inputs to the system (positive int).'''
        raise NotImplementedError

    @property
    def final_time(self):
        '''Time horizon of the system; can be infinite
        (positive float or np.inf).'''
        raise NotImplementedError

    @property
    def parameters(self):
        '''Returns the `ProblemParameters` instance specifying parameters for
        the cost function and system dynamics.'''
        return self._params

    def _update_params(self, **new_params):
        '''Things the subclass does when problem parameters are changed. Also
        called during initialization.'''
        pass

    def distances(self, xa, xb):
        '''
        Calculate the problem-relevant distance metric of a batch of states from
        another state or batch of states. The default metric is Euclidean.

        Parameters
        ----------
        xa : (n_states, n_a) or (n_states,) array
            First batch of points.
        xb : (n_states, n_b) or (n_states,) array
            Second batch of points.

        Returns
        -------
        dist : (n_a, n_b) array
            ||xa - xb|| for each point in xa and xb
        '''
        xa = xa.reshape(self.n_states, -1).T
        xb = xb.reshape(self.n_states, -1).T
        return cdist(xa, xb, metric='euclidean')

    def sample_initial_conditions(self, n_samples=1, **kwargs):
        '''
        Generate initial conditions from the problem's domain of interest.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        kwargs
            Other keyword arguments implemented by the subclass.

        Returns
        -------
        x0 : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x0` will be a one-dimensional array.
        '''
        raise NotImplementedError

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
        raise NotImplementedError

    def running_cost_gradients(self, x, u, return_dLdx=True, return_dLdu=True):
        '''
        Evaluate the gradients of the running cost, dL/dx (x,u) and dL/du (x,u),
        at one or multiple state-control pairs. Default implementation
        approximates this with finite differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdx : bool, default=True
            If True, compute the gradient with respect to states, dL/dx.
        return_dLdu : bool, default=True
            If True, compute the gradient with respect to controls, dL/du.

        Returns
        -------
        dLdx : (n_states,) or (n_states, n_points) array
            State gradients dL/dx (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_controls,) or (n_controls, n_points) array
            Control gradients dL/du (x,u) evaluated at pair(s) (x,u).
        '''
        L = self.running_cost(x, u)

        if return_dLdx:
            dLdx = approx_derivative(
                lambda x: self.running_cost(x, u), x,
                f0=L, method=self._fin_diff_method
            )
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = approx_derivative(
                lambda u: self.running_cost(x, u), u,
                f0=L, method=self._fin_diff_method
            )
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hessians(self, x, u, return_dLdx=True, return_dLdu=True):
        '''
        Evaluate the Hessians of the running cost, d^2L/dx^2 (x,u) and
        d^2L/du^2 (x,u), at one or multiple state-control pairs. Default
        implementation approximates this with finite differences.

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
        dLdx, dLdu = self.running_cost_gradients(x, u)

        if return_dLdx:
            dLdx = approx_derivative(
                lambda x: self.running_cost_gradients(x, u, return_dLdu=False),
                x, f0=dLdx, method=self._fin_diff_method
            )
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = approx_derivative(
                lambda u: self.running_cost_gradients(x, u, return_dLdx=False),
                u, f0=dLdu, method=self._fin_diff_method
            )
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def terminal_cost(self, x):
        '''
        Evaluate the terminal cost F(x) at one or more states.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, sample).

        Returns
        -------
        F : float or (n_points,) array
            Terminal cost(s) F(x) evaluated.
        '''
        raise NotImplementedError

    def total_cost(self, t, x, u):
        '''Computes the accumulated running cost J(t) of a state-control trajectory.'''
        L = self.running_cost(x, u)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate(([0.], J))

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
        raise NotImplementedError

    def jacobians(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        '''
        Evaluate the Jacobians of the dynamics with respect to states and
        controls at single or multiple time instances. Default implementation
        approximates the Jacobians with finite differences.

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
        f0 : (n_states,) or (n_states, n_points) array, optional
            Dynamics evaluated at current state and control pair.

        Returns
        -------
        dfdx : (n_states, n_states) or (n_states, n_states, n_points) array
            Jacobian with respect to states.
        dfdu : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Jacobian with respect to controls.
        '''
        if f0 is None:
            f0 = self.dynamics(x, u)

        # Jacobian with respect to states
        if return_dfdx:
            dfdx = approx_derivative(
                lambda x: self.dynamics(x, u), x,
                f0=f0, method=self._fin_diff_method
            )
            if not return_dfdu:
                return dfdx

        # Jacobian with respect to controls
        if return_dfdu:
            dfdu = approx_derivative(
                lambda u: self.dynamics(x, u), u,
                f0=f0, method=self._fin_diff_method
            )
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def closed_loop_jacobian(self, x, controller):
        '''
        Evaluate the Jacobian of the closed-loop dynamics at single or multiple
        time instances.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        controller : object
            `Controller` instance implementing `__call__` and `jacobian`.

        Returns
        -------
        dfdx : (n_states, n_states) or (n_states, n_states, n_points) array
            Closed-loop Jacobian df/dx + df/du * du/dx.
        '''
        dfdx, dfdu = self.jacobians(x, controller(x))
        dudx = controller.jacobian(x)

        while dfdu.ndim < 3:
            dfdu = dfdu[...,None]
        while dudx.ndim < 3:
            dudx = dudx[...,None]

        dfdx += np.einsum('ijk,jhk->ihk', dfdu, dudx)

        if x.ndim < 2:
            return dfdx[:,:,0]

        return dfdx

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
        raise NotImplementedError

    def optimal_control_jacobian(self, x, p, u0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed. Default implementation uses finite
        differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        u0 : (n_controls,) or (n_controls, n_points) array, optional
            `self.optimal_control(x, p)`, pre-evaluated at the inputs.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, du/dx (x; p=p).
        '''
        p = p.reshape(x.shape)

        return approx_derivative(
            lambda x: self.optimal_control(x, p), x,
            f0=u0, method=self._fin_diff_method
        )

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
        x = xp[:self.n_states]
        p = xp[self.n_states:]
        u = self.optimal_control(x, p)

        # State dynamics
        dxdt = self.dynamics(x, u)

        # Evaluate closed loop Jacobian using chain rule
        dfdx, dfdu = self.jacobians(x, u, f0=dxdt)
        dudx = self.optimal_control_jacobian(x, p, u0=u)

        dfdx += np.einsum('ijk,jhk->ihk', dfdu, dudx)

        dLdx, dLdu = self.running_cost_gradients(x, u)

        if dLdx.ndim < 2:
            dLdx = dLdx[:,None]
        if dLdu.ndim < 2:
            dLdu = dLdu[:,None]

        dLdx += np.einsum('ik,ijk->jk', dLdu, dudx)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdx = dLdx + np.einsum('ijk,ik->jk', dfdx, p)

        return np.vstack((dxdt, -dHdx))

    def make_pontryagin_boundary(self, x0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on costate
        is zero.

        Parameters
        ----------
        x0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of xp_0 (augmented states at initial time) and xp_1
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        x0 = x0.flatten()
        def bc(xp_0, xp_1):
            return np.concatenate((
                xp_0[:self.n_states] - x0, xp_1[self.n_states:]
            ))
        return bc

    def hamiltonian(self, x, u, p):
        '''
        Evaluate the Pontryagin Hamiltonian,
            H(x,u,p) = L(x,u) + <p, f(x,u)>
        where `L(x,u)` is the running cost, `p` is the costate or value
        gradient, and `f(x,u)` is the dynamics. A necessary condition for
        optimality is that `H(x,u,p) = 0` for the whole trajectory.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(x, u)
        f = self.dynamics(x, u)
        return L + np.sum(p * f, axis=0)

    def constraint_fun(self, x):
        '''
        A (vector-valued) function which is zero when the state constraints are
        satisfied.

        Parameters
        ----------
        x : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        -------
        c : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that c(x)=0 means that x satisfies the state
            constraints.
        '''
        return

    def constraint_jacobian(self, x):
        '''
        Constraint function Jacobian dc/dx of self.constraint_fun. Default
        implementation approximates this with finite differences.

        Parameters
        ----------
        x : (n_states,) array
            Current state.

        Returns
        -------
        dcdx : (n_constraints, n_states) array or None
            dc/dx evaluated at the point x, where c(x)=self.constraint_fun(x).
        '''
        c0 = self.constraint_fun(x)
        if c0 is None:
            return

        return approx_derivative(
            self.constraint_fun, x, f0=c0, method=self._fin_diff_method
        )

    def make_integration_events(self):
        '''
        Construct a (list of) callables that are tracked during integration for
        times at which they cross zero. Such events can terminate integration
        early.

        Returns
        -------
        events : None, callable, or list of callables
            Each callable has a function signature e = event(t, x). If the ODE
            integrator finds a sign change in e then it searches for the time t
            at which this occurs. If event.terminal = True then integration
            stops.
        '''
        return

class LinearQuadraticProblem(OptimalControlProblem):
    '''
    Template super class for defining an infinite horizon linear quadratic
    regulator problem. Requires the following parameters upon initialization.

    Parameters
    ----------
    xf : (n_states, 1) array
        Goal state, nominal linearization point.
    uf : (n_controls, 1) array
        Control values at nominal linearization point.
    A : (n_states, n_states) array
        State Jacobian matrix at nominal equilibrium.
    B : (n_states, n_controls) array
        Control Jacobian matrix at nominal equilibrium.
        this with finite differences.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    '''
    _params = ProblemParameters(
        xf=None, uf=None,
        A=None, B=None, Q=None, R=None,
        u_lb=None, u_ub=None
    )

    def _saturate(self, u):
        return saturate(u, self.u_lb, self.u_ub)

    @property
    def n_states(self):
        '''The number of system states (positive int).'''
        if hasattr(self, 'A'):
            return self.A.shape[1]
        else:
            raise RuntimeError('State Jacobian matrix has not been initialized.')

    @property
    def n_controls(self):
        '''The number of control inputs to the system (positive int).'''
        if hasattr(self, 'B'):
            return self.B.shape[1]
        else:
            raise RuntimeError('Control Jacobian matrix has not been initialized.')

    @property
    def final_time(self):
        '''Time horizon of the system.'''
        return np.inf

    def _update_params(self, **new_params):
        if 'A' in new_params or not hasattr(self, 'A'):
            try:
                self.A = np.array(self._params.A)
                self.A = self.A.reshape(self.A.shape[0], self.A.shape[0])
            except:
                raise RuntimeError(
                    'Must initialize state Jacobian `A` with a square array.'
                )

        if 'B' in new_params or not hasattr(self, 'B'):
            try:
                self.B = np.array(self._params.B)
                self.B = self.B.reshape(self.n_states, -1)
            except:
                raise RuntimeError(
                    'Must initialize control Jacobian `B` with an array of '
                    + 'shape `(n_states, n_controls)`.'
                )

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

        self.xf = np.reshape(xf, (-1,1))
        self.uf = np.reshape(uf, (-1,1))

        self.n_states = self.xf.shape[0]
        self.n_controls = self.uf.shape[0]

        # Approximate state matrices numerically if not given
        if A is None or B is None:
            _A, _B = jacobians(self.xf, self.uf, f0=np.zeros_like(self.xf))

        if A is None:
            A = _A
            A[np.abs(A) < 1e-10] = 0.

        if B is None:
            B = _B
            B[np.abs(B) < 1e-10] = 0.

        # Approximate cost matrices numerically if not given
        if Q is None or R is None:
            raise NotImplementedError("need to implement numerical cost Hessians")

        self.A = np.reshape(A, (self.n_states, self.n_states))
        self.B = np.reshape(B, (self.n_states, self.n_controls))
        self.Q = np.reshape(Q, (self.n_states, self.n_states))
        self.R = np.reshape(R, (self.n_controls, self.n_controls))
