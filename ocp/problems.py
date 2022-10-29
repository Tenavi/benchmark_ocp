import numpy as np

from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import sparse

class OptimalControlProblem:
    '''Defines an optimal control problem (OCP).

    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.
    '''
    def __init__(
            self, n_states, n_controls, u_lb=None, u_ub=None, final_time=np.inf
        ):
        self.n_states, self.n_controls = n_states, n_controls

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = np.reshape(self.u_lb, (self.n_controls,1))
        if self.u_ub is not None:
            self.u_ub = np.reshape(self.u_ub, (self.n_controls,1))

        self.final_time = final_time

        self.linearizations = []

    def running_cost(self, x, u):
        '''
        Evaluate the running cost l(x,u) at one or multiple state-control pairs.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) l(x,u) evaluated at pair(s) (x,u).
        '''
        raise NotImplementedError

    def total_cost(self, t, x, u):
        '''Computes the accumulated running cost J(t) of a state-control trajectory.'''
        L = self.running_cost(x, u)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate((J, J[-1:]))

    def running_cost_gradient(self, x, u, dLdx=True, dLdu=True):
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
        dLdx : bool, default=True
            If True, compute the gradient with respect to states, dL/dx.
        dLdu : bool, default=True
            If True,compute the gradient with respect to controls, dL/du.

        Returns
        -------
        dLdx : (n_states,) or (n_states, n_points) array
            State gradients dL/dx (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_states,) or (n_states, n_points) array
            Control gradients dL/du (x,u) evaluated at pair(s) (x,u).
        '''
        L = self.running_cost(x, u)

        if dLdx:
            dLdx = approx_derivative(lambda x: self.running_cost(x, u), x, f0=L)
            if not dLdu:
                return dLdx

        if dLdu:
            dLdu = approx_derivative(lambda u: self.running_cost(x, u), u, f0=L)
            if not dLdx:
                return dLdu

        return dLdx, dLdu

    def Hamiltonian(self, X, U, dVdX):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(X,U,dVdX) = L(X,U) + <dVdX, F(X,U)>
        where L(X,U) is the running cost, dVdX is the costate or value gradient,
        and F(X,U) is the dynamics. A necessary condition for optimality is that
        H(X,U,dVdX) ~ 0 for the whole trajectory.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian each each point in time.
        '''
        L = self.running_cost(X, U)
        F = self.dynamics(X, U)
        return L + np.sum(dVdX * F, axis=0)

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
        raise NotImplementedError

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
        F0 : (n_states,) or (n_states, n_points) array, optional
            Dynamics evaluated at current state and control pair.

        Returns
        -------
        dFdX : (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dX.
        '''
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        if F0 is None:
            F0 = self.dynamics(X, U)

        # Jacobian with respect to states
        def F_wrapper(X_flat):
            X = X_flat.reshape(self.n_states, -1)
            return self.dynamics(X, U).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(X.shape[-1])]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dFdX = approx_derivative(
            F_wrapper, X.flatten(), f0=F0.flatten(), sparsity=sparsity
        )
        dFdX = np.asarray(dFdX[sparsity.nonzero()])
        dFdX = dFdX.reshape(self.n_states, self.n_states, -1)

        # Jacobian with respect to controls
        def F_wrapper(U_flat):
            U = U_flat.reshape(self.n_controls, -1)
            return self.dynamics(X, U).flatten()

        # Make sparsity pattern
        sparsity = sparse.hstack([sparse.identity(X.shape[-1])]*self.n_controls)
        sparsity = sparse.vstack([sparsity]*self.n_states)

        dFdU = approx_derivative(
            F_wrapper, U.flatten(), f0=F0.flatten(), sparsity=sparsity
        )
        dFdU = np.asarray(dFdU[sparsity.nonzero()])
        dFdU = dFdU.reshape(self.n_states, self.n_controls, -1)

        return dFdX, dFdU

    def closed_loop_jacobian(self, X, controller):
        '''
        Evaluate the Jacobian of the closed-loop dynamics at single or multiple
        time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        controller : object
            Controller instance implementing eval_U and eval_dUdX methods.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Closed-loop Jacobian dF/dX + dF/dU * dU/dX.
        '''
        dFdX, dFdU = self.jacobians(X, controller.eval_U(X))
        dUdX = controller.eval_dUdX(X)

        while dFdU.ndim < 3:
            dFdU = dFdU[...,None]
        while dUdX.ndim < 3:
            dUdX = dUdX[...,None]

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        if X.ndim < 2:
            dFdX = np.squeeze(dFdX)

        return dFdX

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
        raise NotImplementedError

    def jac_U_star(self, X, dVdX, U0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed. Default implementation uses finite
        differences.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : (n_controls,) or (n_controls, n_points) array, optional
            U_star(X, dVdX), pre-evaluated at the inputs.

        Returns
        -------
        dUdX : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, dU/dX (X; dVdX).
        '''
        if U0 is None:
            U0 = self.U_star(X, dVdX)

        dVdX = dVdX.reshape(self.n_states, -1)

        # Numerical derivative of optimal feedback policy
        def U_wrapper(X_flat):
            X = X_flat.reshape(self.n_states, -1)
            return self.U_star(X, dVdX).flatten()

        # Make sparsity pattern
        sparsity = sparse.identity(dVdX.shape[-1])
        sparsity = sparse.hstack([sparsity]*self.n_states)
        sparsity = sparse.vstack([sparsity]*self.n_controls)

        dUdX = approx_derivative(
            U_wrapper, X.flatten(), f0=U0.flatten(), sparsity=sparsity
        )
        dUdX = np.asarray(dUdX[sparsity.nonzero()])
        return dUdX.reshape(self.n_controls, self.n_states, -1)

    def bvp_dynamics(self, t, X_aug):
        '''
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.
        Default implementation uses finite differences for the costate dynamics.

        Parameters
        ----------
        t : (n_points,) array
            Time collocation points for each state.
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        X = X_aug[:self.n_states]
        dVdX = X_aug[self.n_states:2*self.n_states]
        U = self.U_star(X, dVdX)

        # State dynamics
        dXdt = self.dynamics(X, U)

        # Evaluate closed loop Jacobian using chain rule
        dFdX, dFdU = self.jacobians(X, U, F0=dXdt)
        dUdX = self.jac_U_star(X, dVdX, U0=U)

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        # Lagrangian and Lagrangian gradient
        L = np.atleast_2d(self.running_cost(X, U))
        dLdx, dLdu = self.running_cost_gradient(X, U)

        if dLdx.ndim < 2:
            dLdx = dLdx[:,None]
        if dLdu.ndim < 2:
            dLdu = dLdu[:,None]

        dLdx += np.einsum('ik,ijk->jk', dLdu, dUdX)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdX = dLdx + np.einsum('ijk,ik->jk', dFdX, dVdX)

        return np.vstack((dXdt, -dHdX, -L))

    def make_bc(self, X0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on lambda is
        zero.

        Parameters
        ----------
        X0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of X_aug_0 (augmented states at initial time) and X_aug_T
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        X0 = X0.flatten()
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((
                X_aug_0[:self.n_states] - X0, X_aug_T[self.n_states:]
            ))
        return bc

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
        ----------
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

        X0 = np.random.rand(self.n_states, Ns)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb

        if dist is not None:
            X0 = X0 - self.xf
            X0_norm = dist / np.linalg.norm(X0, 1, axis=0)
            X0 = X0_norm * X0 + self.xf

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def apply_state_constraints(self, X):
        '''
        Manually update states to satisfy some state constraints. At present
        time, the OCP format only supports constraints which are intrinsic to
        the dynamics (such as quaternions or periodicity), not dynamic
        constraints which need to be satisfied by admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states with constrained values.
        '''
        return X

    def constraint_fun(self, X):
        '''
        A (vector-valued) function which is zero when the state constraints are
        satisfied. At present time, the OCP format only supports constraints
        which are intrinsic to the dynamics (such as quaternions or
        periodicity), not dynamic constraints which need to be satisfied by
        admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        return

    def constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun. Default
        implementation approximates this with central differences.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        dCdX : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        C0 = self.constraint_fun(X)
        if C0 is None:
            return

        return approx_derivative(self.constraint_fun, X, f0=C0)

    def make_integration_events(self):
        '''
        Construct a (list of) callables that are tracked during integration for
        times at which they cross zero. Such events can terminate integration
        early.

        Returns
        -------
        events : None, callable, or list of callables
            Each callable has a function signature e = event(t, X). If the ODE
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
            _A, _B = jacobians(self.xf, self.uf, F0=np.zeros_like(self.xf))

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
