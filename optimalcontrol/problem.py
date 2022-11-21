import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import sparse

class ProblemParameters:
    '''Utility class to store cost function and system dynamics parameters.'''
    def __init__(self, **params):
        self.__dict__.update(params)

    def update(self, **params):
        '''Modify individual or multiple parameters using keyword arguments.'''
        self.__dict__.update(params)

class OptimalControlProblem:
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.
    '''
    # Default cost and system parameters. Should be overwritten by subclass.
    _params = ProblemParameters()

    def __init__(self, **problem_parameters):
        '''Initialize the OCP with some cost and system parameters.

        Parameters
        ----------
        problem_parameters : dict, default={}
            Parameters specifying the cost function and system dynamics. If
            empty, defaults defined by the subclass will be used.
        '''
        if not isinstance(self._params, ProblemParameters):
            self._params = ProblemParameters(**self._params)
        self.update_parameters(**problem_parameters)

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

    def update_parameters(self, **new_params):
        '''Modify cost function and dynamics parameters using keyword arguments.
        This is the preferred way to change problem parameters since the OCP may
        have to make other calculations based on new parameters.'''
        self._params.update(**new_params)
        self._update_params(**new_params)

    def _update_params(self, **new_params):
        '''Things the subclass does when problem parameters are changed.'''
        pass

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
        raise NotImplementedError

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
        L = self.running_cost(x, u)

        if return_dLdx:
            dLdx = approx_derivative(lambda x: self.running_cost(x, u), x, f0=L)
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = approx_derivative(lambda u: self.running_cost(x, u), u, f0=L)
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

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

    def jacobians(self, x, u, f0=None):
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
        f0 : (n_states,) or (n_states, n_points) array, optional
            Dynamics evaluated at current state and control pair.

        Returns
        -------
        dfdx : (n_states, n_states, n_points) array
            Jacobian with respect to states.
        dfdu : (n_states, n_controls, n_points) array
            Jacobian with respect to controls.
        '''
        x = x.reshape(self.n_states, -1)
        u = u.reshape(self.n_controls, -1)

        if f0 is None:
            f0 = self.dynamics(x, u)
        f0 = f0.flatten()

        # Jacobian with respect to states
        def f_wrapper(x_flat):
            x = x_flat.reshape(self.n_states, -1)
            return self.dynamics(x, u).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(x.shape[-1])]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dfdx = approx_derivative(
            f_wrapper, x.flatten(), f0=f0, sparsity=sparsity
        )
        dfdx = np.asarray(dfdx[sparsity.nonzero()])
        dfdx = dfdx.reshape(self.n_states, self.n_states, -1)

        # Jacobian with respect to controls
        def f_wrapper(u_flat):
            u = u_flat.reshape(self.n_controls, -1)
            return self.dynamics(x, u).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(x.shape[-1])]*self.n_controls)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dfdu = approx_derivative(
            f_wrapper, u.flatten(), f0=f0, sparsity=sparsity
        )
        dfdu = np.asarray(dfdu[sparsity.nonzero()])
        dfdu = dfdu.reshape(self.n_states, self.n_controls, -1)

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
            BaseController instance implementing `__call__` and `jacobian`.

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
        raise NotImplementedError

    def optimal_control_jac(self, x, dVdx, u0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed. Default implementation uses finite
        differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdx : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        u0 : (n_controls,) or (n_controls, n_points) array, optional
            self.optimal_control(x, dVdx), pre-evaluated at the inputs.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, du/dx (x; dVdx).
        '''
        if u0 is None:
            u0 = self.optimal_control(x, dVdx)

        dVdx = dVdx.reshape(self.n_states, -1)

        # Numerical derivative of optimal feedback policy
        def u_wrapper(x_flat):
            x = x_flat.reshape(self.n_states, -1)
            return self.optimal_control(x, dVdx).flatten()

        # Make sparsity pattern
        sparsity = sparse.identity(dVdx.shape[-1])
        sparsity = sparse.hstack([sparsity]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_controls)

        dudx = approx_derivative(
            u_wrapper, x.flatten(), f0=u0.flatten(), sparsity=sparsity
        )
        dudx = np.asarray(dudx[sparsity.nonzero()])
        dudx = dudx.reshape(self.n_controls, self.n_states, -1)

        if x.ndim < 2:
            return dudx[:,:,0]

        return dudx

    def bvp_dynamics(self, t, xp):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

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
        x = xp[:self.n_states]
        p = xp[self.n_states:]
        u = self.optimal_control(x, p)

        # State dynamics
        dxdt = self.dynamics(x, u)

        # Evaluate closed loop Jacobian using chain rule
        dfdx, dfdu = self.jacobians(x, u, f0=dxdt)
        dudx = self.optimal_control_jac(x, p, u0=u)

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

    def hamiltonian(self, x, u, dVdx):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(x,u,dVdx) = L(x,u) + <dVdx, f(x,u)>
        where L(x,u) is the running cost, dVdx is the costate or value gradient,
        and f(x,u) is the dynamics. A necessary condition for optimality is that
        H(x,u,dVdx) = 0 for the whole trajectory.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdx : (n_states,) or (n_states, n_points) array
            Value gradient dV/dx (x,u) evaluated at pair(s) (x,u).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(x, u)
        f = self.dynamics(x, u)
        return L + np.sum(dVdx * f, axis=0)

    def norm(self, x, xf=None):
        '''
        Calculate the distance of a batch of spatial points from xf or zero.
        By default uses L2 norm.

        Parameters
        ----------
        x : (n_states, n_data) or (n_states,) array
            Points to compute distances for
        xf : (n_states,) array, optional
            If provided, calculate ||x - xf||

        Returns
        -------
        x_norm : (n_data,) array
            Norm for each point in x
        '''
        x = x.reshape(self.n_states, -1)
        if xf is not None:
            x = x - xf.reshape(self.n_states, 1)
        return np.linalg.norm(x, axis=0)

    def sample_initial_condition(self, Ns, dist=None):
        '''Uniform sampling from the initial condition domain.'''
        raise NotImplementedError

        x0 = np.random.rand(self.n_states, Ns)
        x0 = (self.x0_ub - self.x0_lb) * x0 + self.x0_lb

        if dist is not None:
            x0 = x0 - self.xf
            x0_norm = dist / np.linalg.norm(x0, 1, axis=0)
            x0 = x0_norm * x0 + self.xf

        if Ns == 1:
            x0 = x0.flatten()
        return x0

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
        implementation approximates this with central differences.

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

        return approx_derivative(self.constraint_fun, x, f0=c0)

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

class LinearProblem:
    def __init__(self, xf, uf, A=None, B=None, Q=None, R=None, jacobians=None):
        '''
        Parameters
        ----------
        xf : (n_states, 1) array
            Goal state, nominal linearization point.
        uf : (n_controls, 1) array
            Control values at nominal linearization point.
        A : (n_states, n_states) array or None
            State Jacobian matrix at nominal equilibrium. If None, approximates
            this with central differences.
        B : (n_states, n_controls) array or None
            Control Jacobian matrix at nominal equilibrium. If None, approximates
            this with central differences.
        Q : (n_states, n_states) array
            Hessian of running cost with respect to states. Must be positive
            semi-definite.
        R : (n_controls, n_controls) array
            Hessian of running cost with respect to controls. Must be positive
            definite.
        jacobians : callable, optional
        '''
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
