import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.spatial.distance import cdist

from .parameters import ProblemParameters
from .sampling import UniformSampler
from . import utilities as utils


class OptimalControlProblem:
    """
    Template super class defining an optimal control problem (OCP) including
    dynamics, running cost, and optimal control as a function of costate.
    """
    # Dicts of required and optional cost and system parameters.
    # Parameters without default values should have None entries.
    # To be overwritten by subclass implementations.
    _required_parameters = {}
    _optional_parameters = {}
    # Finite difference method for default gradient, Jacobian, and Hessian
    # approximations
    _fin_diff_method = "3-point"

    def __init__(self, **problem_parameters):
        """
        Parameters
        ----------
        problem_parameters : dict, default={}
            Parameters specifying the cost function and system dynamics. If
            empty, defaults defined by the subclass will be used.
        """
        self._params = ProblemParameters(
            required=self._required_parameters.keys(),
            optional=self._optional_parameters.keys(),
            update_fun=self._update_params)
        problem_parameters = {**self._required_parameters,
                              **self._optional_parameters,
                              **problem_parameters}
        self.parameters.update(**problem_parameters)

    @property
    def n_states(self):
        """The number of system states (positive int)."""
        raise NotImplementedError

    @property
    def n_controls(self):
        """The number of control inputs to the system (positive int)."""
        raise NotImplementedError

    @property
    def final_time(self):
        """Time horizon of the system; can be infinite (positive float or
        np.inf)."""
        raise NotImplementedError

    def _reshape_inputs(self, x, u):
        """
        Utility function to reshape 1d array state and controls into 2d arrays.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        x : (n_states, n_points) array
            State(s) arranged by (dimension, time). If the input was flat,
            `n_points = 1`.
        u : (n_controls, n_points) array
            Control(s) arranged by (dimension, time). If the input was flat,
            `n_points = 1`.
        squeeze: bool
            True if either input was flat.

        Raises
        ------
        ValueError
            If cannot reshape states and controls to the correct sizes, or if
            `x.shape[1] != u.shape[1]`.
        """
        squeeze = np.ndim(x) < 2 or np.ndim(u) < 2

        if np.ndim(x) != 2 or np.shape(x)[0] != self.n_states:
            try:
                x = np.reshape(x, (self.n_states, -1))
            except:
                raise ValueError(
                    "x must be an array of shape (n_states,) or (n_states,n_points)"
                )
        elif not isinstance(x, np.ndarray):
            x = np.array(x)

        if np.ndim(u) != 2 or np.shape(u)[0] != self.n_controls:
            try:
                u = np.reshape(u, (self.n_controls, -1))
            except:
                raise ValueError(
                    "u must be an array of shape (n_controls,) or (n_controls,n_points)"
                )
        elif not isinstance(u, np.ndarray):
            u = np.array(u)

        if x.shape[1] != u.shape[1]:
            raise ValueError(
                "x.shape[1] = %d != u.shape[1] = %d" % (x.shape[1], u.shape[1])
            )

        return x, u, squeeze

    @property
    def parameters(self):
        """Returns a `ProblemParameters` instance specifying parameters for the
        cost function(s) and system dynamics."""
        return self._params

    def _update_params(self, obj, **new_params):
        """
        Things the subclass does when problem parameters are changed. Also
        called during initialization.

        Parameters
        ----------
        obj : ProblemParameters instance
            Pass an instance of `ProblemParameters` to modify its instance
            attributes, if needed.
        **new_params : dict
            Parameters which are changing.
        """
        pass

    def sample_initial_conditions(self, n_samples=1, **kwargs):
        """
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
        """
        raise NotImplementedError

    def distances(self, xa, xb):
        """
        Calculate the problem-relevant distance of a batch of states from
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
            ||xa - xb|| for each point in xa and xb.
        """
        xa = np.reshape(xa, (self.n_states, -1)).T
        xb = np.reshape(xb, (self.n_states, -1)).T
        return cdist(xa, xb, metric="euclidean")

    def running_cost(self, x, u):
        """
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
        """
        raise NotImplementedError

    def running_cost_gradients(self, x, u, return_dLdx=True, return_dLdu=True,
                               L0=None):
        """
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
        L0 : (1,) or (n_points,) array, optional
            Running cost evaluated at current state and control pair.

        Returns
        -------
        dLdx : (n_states,) or (n_states, n_points) array
            State gradients dL/dx (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_controls,) or (n_controls, n_points) array
            Control gradients dL/du (x,u) evaluated at pair(s) (x,u).
        """
        if L0 is None:
            L0 = self.running_cost(x, u)

        if return_dLdx:
            dLdx = utils.approx_derivative(lambda x: self.running_cost(x, u), x,
                f0=L0, method=self._fin_diff_method)
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = utils.approx_derivative(lambda u: self.running_cost(x, u), u,
                f0=L0, method=self._fin_diff_method)
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hessians(self, x, u, return_dLdx=True, return_dLdu=True,
                              L0=None):
        """
        Evaluate the Hessians of the running cost, $d^2L/dx^2 (x,u)$ and
        $d^2L/du^2 (x,u)$, at one or multiple state-control pairs. Default
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
        L0 : (1,) or (n_points,) array, optional
            Running cost evaluated at current state and control pair.

        Returns
        -------
        dLdx : (n_states, n_states) or (n_states, n_states, n_points) array
            State Hessians dL^2/dx^2 (x,u) evaluated at pair(s) (x,u).
        dLdu : (n_controls,) or (n_controls, n_controls, n_points) array
            Control Hessians dL^2/du^2 (x,u) evaluated at pair(s) (x,u).
        """
        if L0 is None:
            L0 = self.running_cost(x, u)

        dLdx, dLdu = self.running_cost_gradients(x, u, L0=L0)

        if return_dLdx:
            dLdx = utils.approx_derivative(
                lambda x: self.running_cost_gradients(x, u, return_dLdu=False,
                                                      L0=L0),
                x, f0=dLdx, method=self._fin_diff_method)
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = utils.approx_derivative(
                lambda u: self.running_cost_gradients(x, u, return_dLdx=False,
                                                      L0=L0),
                u, f0=dLdu, method=self._fin_diff_method)
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def terminal_cost(self, x):
        """
        Evaluate the terminal cost F(x) at one or more states.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, sample).

        Returns
        -------
        F : float or (n_points,) array
            Terminal cost(s) F(x) evaluated.
        """
        raise NotImplementedError

    def total_cost(self, t, x, u):
        """
        Computes the accumulated running cost, J(t), of a state-control
        trajectory, using trapezoidal integration.

        Parameters
        ----------
        t : (n_points,) array
            Time values at which state and control vectors are given.
        x : (n_states, n_points) array
            Time series of system states.
        u : (n_controls, n_points) array
            Time series of control inputs.

        Returns
        -------
        J : (n_points,) array
            Integral of the running cost over time.
        """
        L = self.running_cost(x, u)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate(([0.], J))

    def dynamics(self, x, u):
        """
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
        """
        raise NotImplementedError

    def jacobians(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        """
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
        """
        if f0 is None:
            f0 = self.dynamics(x, u)

        # Jacobian with respect to states
        if return_dfdx:
            dfdx = utils.approx_derivative(
                lambda x: self.dynamics(x, u), x,
                f0=f0, method=self._fin_diff_method
            )
            if not return_dfdu:
                return dfdx

        # Jacobian with respect to controls
        if return_dfdu:
            dfdu = utils.approx_derivative(
                lambda u: self.dynamics(x, u), u,
                f0=f0, method=self._fin_diff_method
            )
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        """
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
        """
        raise NotImplementedError

    def optimal_control_jacobian(self, x, p, u0=None):
        """
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
        """
        p = np.reshape(p, np.shape(x))

        return utils.approx_derivative(
            lambda x: self.optimal_control(x, p), x,
            f0=u0, method=self._fin_diff_method
        )

    def bvp_dynamics(self, t, xp):
        """
        Evaluate the augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        t : (n_points,) array
            Time collocation points for each state.
        xp : (2*n_states + 1, n_points) array
            Current state, costate, and value function.

        Returns
        -------
        dxpdt : (2*n_states + 1, n_points) array
            Concatenation of dynamics $dx/dt = f(x,u^*)$, costate dynamics,
            $dp/dt = -dH/dx(x,u^*,p)$, and running cost $L(x,u^*)$, where
            $u^* = u^*(x,p)$ is the optimal control.
        """
        x = xp[:self.n_states]
        p = xp[self.n_states:-1]
        u = self.optimal_control(x, p)

        # State dynamics
        dxdt = self.dynamics(x, u)

        # Evaluate closed loop Jacobian using chain rule
        dfdx, dfdu = self.jacobians(x, u, f0=dxdt)
        dudx = self.optimal_control_jacobian(x, p, u0=u)

        dfdx += np.einsum("ijk,jhk->ihk", dfdu, dudx)

        L = self.running_cost(x, u)
        dLdx, dLdu = self.running_cost_gradients(x, u, L0=L)

        if dLdx.ndim < 2:
            dLdx = dLdx[:,None]
        if dLdu.ndim < 2:
            dLdu = dLdu[:,None]

        dLdx += np.einsum("ik,ijk->jk", dLdu, dudx)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdx = dLdx + np.einsum("ijk,ik->jk", dfdx, p)

        return np.vstack((dxdt, -dHdx, -L))

    def hamiltonian(self, x, u, p):
        """
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
        """
        L = self.running_cost(x, u)
        f = self.dynamics(x, u)
        return L + np.sum(p * f, axis=0)

    def constraint_fun(self, x):
        """
        A (vector-valued) function which is zero when the state constraints are
        satisfied.

        TODO: Replace with a @property returning a list of
            scipy.optimize.Constraint objects, or something else compatible with
            constrained optimization

        Parameters
        ----------
        x : (n_states, n_points) or (n_states,) array
            Current states.

        Returns
        -------
        c : (n_constraints,) or (n_constraints, n_points) array or None
            Algebraic equation such that c(x)=0 means that x satisfies the state
            constraints.
        """
        return

    def constraint_jacobian(self, x):
        """
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
        """
        c0 = self.constraint_fun(x)
        if c0 is None:
            return

        return utils.approx_derivative(
            self.constraint_fun, x, f0=c0, method=self._fin_diff_method
        )

    @property
    def integration_events(self):
        """
        Get a (list of) callables that are tracked during integration for times
        at which they cross zero. Such events can terminate integration early.

        Returns
        -------
        events : None, callable, or list of callables
            Each callable has a function signature `e = event(t, x)`. If the ODE
            integrator finds a sign change in `e` then it searches for the time
            `t` at which this occurs. If `event.terminal = True` then
            integration stops.
        """
        return


class LinearQuadraticProblem(OptimalControlProblem):
    """
    Template super class for defining an infinite horizon linear quadratic
    regulator problem. Takes the following parameters upon initialization.

    Parameters
    ----------
    A : (n_states, n_states) array
        State Jacobian matrix at nominal equilibrium.
    B : (n_states, n_controls) array
        Control Jacobian matrix at nominal equilibrium.
    Q : (n_states, n_states) array
        Hessian of running cost with respect to states. Must be positive
        semi-definite.
    R : (n_controls, n_controls) array
        Hessian of running cost with respect to controls. Must be positive
        definite.
    xf : {(n_states, 1) array, float}, default=0.
        Goal state, nominal linearization point. If float, will be broadcast
        into an array of shape `(n_states, 1)`.
    uf : {(n_controls, 1) array, float}, default=0.
        Control values at nominal linearization point. If float, will be
        broadcast into an array of shape `(n_controls, 1)`.
    x0_lb : {(n_states, 1) array, float}
        Lower bounds for hypercube from which to sample initial conditions `x0`.
        If float, will be broadcast into an array of shape `(n_states, 1)`.
    x0_ub : {(n_states, 1) array, float}
        Upper bounds for hypercube from which to sample initial conditions `x0`.
        If float, will be broadcast into an array of shape `(n_states, 1)`.
    u_lb : {(n_controls, 1) array, float}, optional
        Lower control bounds. If float, will be broadcast into an array of shape
        `(n_controls, 1)`.
    u_ub : {(n_controls, 1) array, float}, optional
        Upper control bounds. If float, will be broadcast into an array of shape
        `(n_controls, 1)`.
    x0_sample_seed : int, optional
        Random seed to use for sampling initial conditions.
    """
    _required_parameters = {
        "A": None, "B": None, "Q": None, "R": None, "x0_lb": None, "x0_ub": None
    }
    _optional_parameters = {
        "xf": 0., "uf": 0., "u_lb": None, "u_ub": None, "x0_sample_seed": None
    }

    def _saturate(self, u):
        return utils.saturate(u, self.u_lb, self.u_ub)

    @property
    def n_states(self):
        try:
            return self.parameters.A.shape[1]
        except:
            raise RuntimeError("State Jacobian matrix A has not been initialized.")

    @property
    def n_controls(self):
        try:
            return self.parameters.B.shape[1]
        except:
            raise RuntimeError("Control Jacobian matrix B has not been initialized.")

    @property
    def final_time(self):
        return np.inf

    def _update_params(self, obj, **new_params):
        if "A" in new_params:
            try:
                obj.A = np.atleast_1d(obj.A)
                obj.A = obj.A.reshape(obj.A.shape[0], obj.A.shape[0])
            except:
                raise ValueError("State Jacobian matrix A must have shape (n_states, n_states)")

        if "B" in new_params:
            try:
                obj.B = np.asarray(obj.B)
                if obj.B.ndim == 2 and obj.B.shape[0] != self.n_states:
                    raise
                else:
                    obj.B = np.reshape(obj.B, (self.n_states, -1))
            except:
                raise ValueError("Control Jacobian matrix B must have shape (n_states, n_controls)")

        if "Q" in new_params:
            try:
                obj.Q = np.reshape(obj.Q, (self.n_states, self.n_states))
                eigs = np.linalg.eigvals(obj.Q)
                if not np.all(eigs >= 0.) or not np.allclose(obj.Q, obj.Q.T):
                    raise
                self.singular_Q = np.any(np.isclose(eigs, 0.))
            except:
                raise ValueError("State cost matrix Q must have shape (n_states, n_states) and be positive semi-definite")

        if "R" in new_params:
            try:
                obj.R = np.reshape(obj.R, (self.n_controls, self.n_controls))
                eigs = np.linalg.eigvals(obj.R)
                if not np.all(eigs > 0.) or not np.allclose(obj.R, obj.R.T):
                    raise
            except:
                raise ValueError("Control cost matrix R must have shape (n_controls, n_controls) and be positive definite")

        if "xf" in new_params and obj.A is not None:
            obj.xf = utils.resize_vector(obj.xf, self.n_states)

        if "uf" in new_params and obj.B is not None:
            obj.uf = utils.resize_vector(obj.uf, self.n_controls)

        for key in ("u_lb", "u_ub"):
            if key in new_params or not hasattr(self, key):
                if getattr(obj, key, None) is not None:
                    u_bound = utils.resize_vector(new_params[key], self.n_controls)
                    setattr(obj, key, u_bound)
                setattr(self, key, getattr(obj, key))

        for key in ("A", "B", "Q", "R", "xf", "uf"):
            if key in new_params:
                setattr(self, key, getattr(obj, key))

        if "B" in new_params or "R" in new_params:
            self.RB2 = np.linalg.solve(self.R, self.B.T) / 2.

        if "Q" in new_params or not hasattr(self, "_x0_sampler"):
            self._x0_sampler = UniformSampler(
                lb=obj.x0_lb,
                ub=obj.x0_ub,
                xf=self.xf,
                norm=1 if self.singular_Q else self.Q,
                seed=getattr(obj, "x0_sample_seed", None)
            )
        elif any([
                "x0_lb" in new_params,
                "x0_ub" in new_params,
                "x0_sample_seed" in new_params,
                "xf" in new_params
            ]):
            self._x0_sampler.update(
                lb=new_params.get("x0_lb"),
                ub=new_params.get("x0_ub"),
                xf=self.xf,
                seed=new_params.get("x0_sample_seed")
            )

    def _center_inputs(self, x, u):
        """
        Wrapper of `_reshape_inputs` that reshapes 1d array state and controls
        into 2d arrays, saturates the controls, and subtracts nominal states and
        controls.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        x - xf : (n_states, n_points) array
            State(s) arranged by (dimension, time). If the input was flat,
            `n_points = 1`.
        u - uf : (n_controls, n_points) array
            Control(s) arranged by (dimension, time). If the input was flat,
            `n_points = 1`.
        squeeze: bool
            True if either input was flat.

        Raises
        ------
        ValueError
            If cannot reshape states and controls to the correct sizes, or if
            `x.shape[1] != u.shape[1]`.
        """
        x, u, squeeze = self._reshape_inputs(x, u)
        return x - self.xf, self._saturate(u) - self.uf, squeeze

    def sample_initial_conditions(self, n_samples=1, distance=None):
        """
        Generate initial conditions uniformly from a hypercube, or on the
        surface of a hyperellipse defined by `self.Q`.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance of samples from `self.xf`. If `self.Q` is positive
            definite, the distance is defined by the norm
                `||x|| = sqrt(x.T @ self.Q @ x)`,
            otherwise the l1 norm is used.

        Returns
        -------
        x0 : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x0` will be a one-dimensional array.
        """
        return self._x0_sampler(n_samples=n_samples, distance=distance)

    def distances(self, xa, xb):
        """
        Calculate the distance of a batch of states from another state or batch
        of states. The distance is defined as
            `||xa - xb|| = sqrt((xa - xb).T @ self.Q @ (xa - xb))`.

        Parameters
        ----------
        xa : (n_states, n_a) or (n_states,) array
            First batch of points.
        xb : (n_states, n_b) or (n_states,) array
            Second batch of points.

        Returns
        -------
        dist : (n_a, n_b) array
            ||xa - xb|| for each point in xa and xb.
        """
        xa = np.reshape(xa, (self.n_states, -1)).T
        xb = np.reshape(xb, (self.n_states, -1)).T

        return cdist(xa, xb, metric="mahalanobis", VI=self.Q)

    def running_cost(self, x, u):
        x, u, squeeze = self._center_inputs(x, u)

        # Batch multiply (x - xf).T @ Q @ (x - xf)
        L = np.einsum("ij,ij->j", x, self.Q @ x)

        # Batch multiply (u - uf).T @ R @ (u - xf) and sum
        L += np.einsum("ij,ij->j", u, self.R @ u)

        if squeeze:
            return L[0]

        return L

    def running_cost_gradients(self, x, u, return_dLdx=True, return_dLdu=True,
                               L0=None):
        x, u, squeeze = self._center_inputs(x, u)

        if return_dLdx:
            dLdx = 2. * np.einsum("ij,jb->ib", self.Q, x)
            if squeeze:
                dLdx = dLdx[...,0]
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = 2. * np.einsum("ij,jb->ib", self.R, u)
            if squeeze:
                dLdu = dLdu[...,0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hessians(self, x, u, return_dLdx=True, return_dLdu=True,
                              L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            dLdx = 2. * self.Q
            if not squeeze:
                dLdx = np.tile(dLdx[..., None], (1, 1, x.shape[1]))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = 2. * self.R
            dLdu = np.tile(dLdu[...,None], (1, 1, u.shape[1]))

            # Where the control is saturated, the gradient is constant so the
            # Hessian is zero
            zero_idx = utils.find_saturated(u, min=self.u_lb, max=self.u_ub)
            dLdu[:, zero_idx] = 0.

            if squeeze:
                dLdu = dLdu[...,0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def dynamics(self, x, u):
        x, u, squeeze = self._center_inputs(x, u)

        dxdt = np.matmul(self.A, x) + np.matmul(self.B, u)

        if squeeze:
            return dxdt.flatten()

        return dxdt

    def jacobians(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dfdx:
            dfdx = np.copy(self.A)
            if not squeeze:
                dfdx = np.tile(dfdx[...,None], (1,1,x.shape[1]))
            if not return_dfdu:
                return dfdx

        if return_dfdu:
            dfdu = np.tile(self.B[...,None], (1,1,u.shape[1]))

            # Where the control is saturated, the Jacobian is zero
            zero_idx = utils.find_saturated(u, min=self.u_lb, max=self.u_ub)
            dfdu[:,zero_idx] = 0.

            if squeeze:
                dfdu = dfdu[...,0]
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        p = np.reshape(p, (self.n_states, -1))
        u = self.uf - np.matmul(self.RB2, p)
        u = self._saturate(u)

        if np.ndim(x) < 2:
            return u.flatten()

        return u

    def optimal_control_jacobian(self, x, p, u0=None):
        return np.zeros((self.n_controls, self.n_states) + np.shape(p)[1:])
