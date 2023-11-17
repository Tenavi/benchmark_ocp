import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

from optimalcontrol.utilities import approx_derivative, saturate


class OptimalControlProblem:
    """
    Template superclass defining an optimal control problem (OCP) including
    nonlinear dynamics, cost functions, constraints, and costate dynamics, as
    well as optimal control as a function of state and costate.
    """
    # Dicts of default cost function and dynamics parameters, separated into
    # required and optional parameters. To be overwritten by subclass
    # implementations.
    _required_parameters = {}
    _optional_parameters = {}
    # Finite difference method for default gradient, Jacobian, and Hessian
    # approximations
    _fin_diff_method = '3-point'

    def __init__(self, **problem_parameters):
        """
        Parameters
        ----------
        problem_parameters : dict, default={}
            Parameters specifying the cost function and system dynamics. If
            empty, defaults defined by the subclass will be used.
        """
        # Combine all default and non-default parameters
        problem_parameters = {**self._required_parameters,
                              **self._optional_parameters,
                              **problem_parameters}
        # Initialize the parameter container
        # type(self) is used here in case subclass implementations forget to
        # make _parameter_update_fun a staticmethod.
        self.parameters = ProblemParameters(
            required=self._required_parameters.keys(),
            update_fun=type(self)._parameter_update_fun)
        """`ProblemParameters`. Cost function and system dynamics parameters."""
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
        `np.inf`)."""
        raise NotImplementedError

    @property
    def control_lb(self):
        """(`n_controls`,) array or None. Lower bounds on controls `u`."""
        return getattr(self.parameters, 'u_lb', None)

    @property
    def control_ub(self):
        """(`n_controls`,) array or None. Upper bounds on controls `u`."""
        return getattr(self.parameters, 'u_ub', None)

    @property
    def state_lb(self):
        """(`n_states`,) array or None. Simple lower bound state constraints
        defined by `x >= state_lb`. Currently not compatible with `indirect`
        open loop solvers."""
        return getattr(self.parameters, 'x_lb', None)

    @property
    def state_ub(self):
        """(`n_states`,) array or None. Simple upperr bound state constraints
        defined by `x <= state_ub`. Currently not compatible with `indirect`
        open loop solvers."""
        return getattr(self.parameters, 'x_ub', None)

    def _saturate(self, u):
        """
        Saturate control inputs between lower bound `control_lb` and upper bound
        `control_ub`, if one or both of these are defined.

        Parameters
        ----------
        u : (n_controls,) or (n_controls, n_points) array
            Unsaturated control inputs arranged by (dimension, time).

        Returns
        -------
        u_sat : (n_controls,) or (n_controls, n_points) array
            Saturated control inputs arranged by (dimension, time).
        """
        return saturate(u, lb=self.control_lb, ub=self.control_ub)

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        """
        Performs operations on `self.parameters` during initialization and each
        time `self.parameters.update` is called. This is used for checking
        parameter shapes and performing other needed calculations.

        Parameters
        ----------
        obj : `ProblemParameters`
            In standard use, `obj` refers to `self.parameters`. Note that
            `_parameter_update_fun` allows `obj` itself to be modified.
        **new_params : dict
            Parameters which are being set or changing. In standard use, we can
            expect that each entry of `new_params` is also an attribute of
            `obj`.
        """
        pass

    def sample_initial_conditions(self, n_samples=1, **kwargs):
        """
        Generate initial conditions from the problem's domain of interest.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        **kwargs : dict
            Other keyword arguments implemented by the subclass.

        Returns
        -------
        x0 : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x0` will be a 1d array.
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
            `norm(xa - xb)` for each point in `xa` and `xb`, where the default
            `norm` function is the Euclidean distance.
        """
        xa = np.reshape(xa, (self.n_states, -1)).T
        xb = np.reshape(xb, (self.n_states, -1)).T
        return cdist(xa, xb, metric='euclidean')

    def running_cost(self, x, u):
        """
        Evaluate the running cost `L(x, u)` at one or more state-control pairs.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost `L(x, u)` evaluated at pairs (`x`, `u`).
        """
        raise NotImplementedError

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        """
        Evaluate the gradients of the running cost, $dL/dx (x,u)$ and
        $dL/du (x,u)$, at one or more state-control pairs. Default
        implementation approximates this with finite differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdx : bool, default=True
            If `True`, compute the gradient with respect to states.
        return_dLdu : bool, default=True
            If `True`, compute the gradient with respect to controls.
        L0 : (1,) or (n_points,) array, optional
            Running cost evaluated at state-control pairs (`x`, `u`).

        Returns
        -------
        dLdx : (n_states,) or (n_states, n_points) array
            State gradients $dL/dx (x,u)$ evaluated at pairs (`x`, `u`).
        dLdu : (n_controls,) or (n_controls, n_points) array
            Control gradients $dL/du (x,u)$ evaluated at pairs (`x`, `u`).
        """
        if L0 is None:
            L0 = self.running_cost(x, u)

        if return_dLdx:
            dLdx = approx_derivative(lambda x: self.running_cost(x, u), x,
                                     f0=L0, method=self._fin_diff_method)
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = approx_derivative(lambda u: self.running_cost(x, u), u,
                                     f0=L0, method=self._fin_diff_method)
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        """
        Evaluate the 1/2 times the Hessians of the running cost,
        $1/2 d^2L/dx^2 (x,u)$ and $1/2 d^2L/du^2 (x,u)$, at one or multiple
        state-control pairs. Default implementation approximates this with
        finite differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdx : bool, default=True
            If `True`, compute the Hessian with respect to states.
        return_dLdu : bool, default=True
            If `True`, compute the Hessian with respect to controls.
        L0 : (1,) or (n_points,) array, optional
            Running cost evaluated at state-control pairs (`x`, `u`).

        Returns
        -------
        dLdx : (n_states, n_states) or (n_states, n_states, n_points) array
            1/2 times the state Hessians $1/2 dL^2/dx^2 (x,u)$ evaluated at
            pairs (`x`, `u`).
        dLdu : (n_controls,) or (n_controls, n_controls, n_points) array
            1/2 times the control Hessians $1/2 dL^2/du^2 (x,u)$ evaluated at
            pairs (`x`, `u`).
        """
        if L0 is None:
            L0 = self.running_cost(x, u)

        dLdx, dLdu = self.running_cost_grad(x, u, L0=L0)

        if return_dLdx:
            g = lambda x: self.running_cost_grad(x, u, return_dLdu=False, L0=L0)
            dLdx = 0.5 * approx_derivative(g, x, f0=dLdx,
                                           method=self._fin_diff_method)
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            g = lambda u: self.running_cost_grad(x, u, return_dLdx=False, L0=L0)
            dLdu = 0.5 * approx_derivative(g, u, f0=dLdu,
                                           method=self._fin_diff_method)
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def terminal_cost(self, x):
        """
        Evaluate the terminal cost $F(x)$ at one or more states.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, sample).

        Returns
        -------
        F : (1,) or (n_points,) array
            Terminal cost(s) $F(x)$.
        """
        raise NotImplementedError

    def total_cost(self, t, x, u):
        """
        Computes the accumulated running cost as a function of time, $J(t)$,
        given a state-control trajectory, using trapezoidal integration.

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
            Integrated cost at each time `t`.
        """
        L = self.running_cost(x, u).reshape(-1)
        return cumtrapz(L, t, initial=0.)

    def dynamics(self, x, u):
        """
        Evaluate the closed-loop dynamics at one or more time instances.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).

        Returns
        -------
        dxdt : (n_states,) or (n_states, n_points) array
            System dynamics $dx/dt = f(x,u)$ evaluated at pairs (`x`, `u`).
        """
        raise NotImplementedError

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        """
        Evaluate the Jacobians of the dynamics $df/dx (x,u)$ and $df/du (x,u)$,
        at one or more state-control pairs. Default implementation approximates
        the Jacobians with finite differences.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dfdx : bool, default=True
            If `True`, compute the Jacobian with respect to states.
        return_dfdu : bool, default=True
            If `True`, compute the Jacobian with respect to controls.
        f0 : (n_states,) or (n_states, n_points) array, optional
            Dynamics evaluated at state-control pairs (`x`, `u`).

        Returns
        -------
        dfdx : (n_states, n_states) or (n_states, n_states, n_points) array
            State Jacobians $df/dx (x,u)$ evaluated at pairs (`x`, `u`).
        dfdu : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Control Jacobians $df/du (x,u)$ evaluated at pairs (`x`, `u`).
        """
        if f0 is None:
            f0 = self.dynamics(x, u)

        # Jacobian with respect to states
        if return_dfdx:
            dfdx = approx_derivative(lambda x: self.dynamics(x, u), x, f0=f0,
                                     method=self._fin_diff_method)
            if not return_dfdu:
                return dfdx

        # Jacobian with respect to controls
        if return_dfdu:
            dfdu = approx_derivative(lambda u: self.dynamics(x, u), u, f0=f0,
                                     method=self._fin_diff_method)
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def hamiltonian(self, x, u, p, return_dHdu=False):
        """
        Evaluate the Pontryagin Hamiltonian, `H(x,u,p) = L(x,u) + p.T @ f(x,u)`
        where `x` is the state, `u` is the control, `p` is the costate or value
        gradient, `L(x,u)` is the running cost, and `f(x,u)` is the dynamics. A
        necessary condition for optimality is that `hamiltonian(x,u,p) == 0` for
        the whole trajectory.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        return_dHdu : bool, default=False
            If `return_dHdu=True`, compute and return the gradient of the
            Hamiltonian with respect to controls `u`.

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian evaluated at each triplet (`x`, `u`, `p`).
        dHdu : (n_controls,) or (n_controls, n_points) array
            $dH/du$, the gradient of the Hamiltonian with respect to controls
            `u`. Returned only if `return_dHdu=True`.
        """
        L = self.running_cost(x, u)
        f = self.dynamics(x, u)
        H = L + np.sum(p * f, axis=0)

        if return_dHdu:
            dLdu = self.running_cost_grad(x, u, return_dLdx=False, L0=L)
            dfdu = self.jac(x, u, return_dfdx=False, f0=f)
            dHdx = dLdu + np.einsum('ijk,ik->jk', dfdu, p)
            if np.ndim(u) < 2:
                dHdx = dHdx.reshape(-1)
            return H, dHdx

        return H

    def hamiltonian_minimizer(self, x, p, **minimize_opts):
        """
        Evaluate the minimizer of the Hamiltonian (the optimal control) as a
        function of state and costate, $u = argmin H(x, u, p)$.

        The default performs numerical optimization to find `u`. It should be
        overwritten whenever possible.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        p : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        **minimize_opts : dict, optional
            For the default numerical optimization method, keyword arguments to
            pass to `scipy.optimize.minimize`.

        Returns
        -------
        u : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        """
        x = np.asarray(x)
        p = np.asarray(p)

        u_shape = (self.n_controls,) + x.shape[1:]

        u0 = getattr(self.parameters, 'uf', 0.)
        u0 = np.resize(u0, (self.n_controls,))
        u0 = np.tile(u0, u_shape[1:])

        lb = self.control_lb
        ub = self.control_ub
        if lb is None and ub is None:
            bounds = None
        else:
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf

            lb = np.resize(lb, (self.n_controls,))
            ub = np.resize(ub, (self.n_controls,))

            bounds = Bounds(lb=np.tile(lb, u_shape[1:]),
                            ub=np.tile(ub, u_shape[1:]))

        def obj_fun_and_grad(u):
            u = u.reshape(u_shape)
            H, dHdu = self.hamiltonian(x, u, p, return_dHdu=True)
            return H.sum(), dHdu.reshape(-1)

        minimize_res = minimize(obj_fun_and_grad, u0, bounds=bounds, jac=True,
                                **minimize_opts)

        return minimize_res.x.reshape(u_shape)

    def hamiltonian_minimizer_jac(self, x, p, u0=None):
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
            `self.hamiltonian_minimizer(x, p)`, pre-evaluated at `x` and `p`.

        Returns
        -------
        dudx : (n_controls, n_states, n_points) or (n_controls, n_states) array
            Jacobian of the optimal control with respect to states leaving
            costates fixed, $du/dx (x; p=p)$.
        """
        p = np.reshape(p, np.shape(x))
        dudx = approx_derivative(lambda x: self.hamiltonian_minimizer(x, p), x,
                                 f0=u0, method=self._fin_diff_method)
        return dudx

    def bvp_dynamics(self, t, xp):
        """
        Evaluate the combined dynamics of the state $x$, costate $p$, and value
        function $V$, for Pontryagin's Minimum Principle. The default
        implementation makes use of `self.jac` and `self.running_cost_grad`,
        though subclasses may implement more efficient calculations.

        Parameters
        ----------
        t : (n_points,) array or float
            Time collocation points for each state.
        xp : (2 * n_states + 1, n_points) or (2 * n_states + 1,) array
            Current state, costate, and value function, vertically stacked in
            that order.

        Returns
        -------
        dxpdt : (2 * n_states + 1, n_points) or (2 * n_states + 1,) array
            Vertical stack of dynamics $dx/dt = f(x,u)$, costate dynamics
            $dp/dt = -dH/dx(x,u,p)$, and running cost $L(x,u)$, where
            $u = u(x,p)$ is the optimal control and $H(x,u,p)$ is the
            Pontryagin Hamiltonian.
        """
        x = xp[:self.n_states]
        p = xp[self.n_states:-1]
        u = self.hamiltonian_minimizer(x, p)

        # State dynamics
        dxdt = self.dynamics(x, u)

        # Evaluate closed loop Jacobian using chain rule
        dfdx, dfdu = self.jac(x, u, f0=dxdt)
        dudx = self.hamiltonian_minimizer_jac(x, p, u0=u)

        dfdx += np.einsum('ijk,jhk->ihk', dfdu, dudx)

        L = self.running_cost(x, u)
        dLdx, dLdu = self.running_cost_grad(x, u, L0=L)

        if dLdx.ndim < 2:
            dLdx = dLdx[:, None]
        if dLdu.ndim < 2:
            dLdu = dLdu[:, None]

        dLdx += np.einsum('ik,ijk->jk', dLdu, dudx)

        # Costate dynamics (gradient of optimized Hamiltonian)
        dHdx = dLdx + np.einsum('ijk,ik->jk', dfdx, p)

        dxpdt = np.vstack((dxdt, -dHdx, -L))

        if np.ndim(x) < 2:
            return dxpdt[:, 0]

        return dxpdt

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

    def _reshape_inputs(self, x, u):
        """
        Reshape 1d array state and controls into 2d arrays.

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
            `True` if either input was flat.

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
                raise ValueError("x must be an array of shape (n_states,) or "
                                 "(n_states, n_points)")
        else:
            x = np.asarray(x)

        if np.ndim(u) != 2 or np.shape(u)[0] != self.n_controls:
            try:
                u = np.reshape(u, (self.n_controls, -1))
            except:
                raise ValueError("u must be an array of shape (n_controls,) or "
                                 "(n_controls, n_points)")
        else:
            u = np.asarray(u)

        n_x, n_u = x.shape[1], u.shape[1]
        if n_x != n_u:
            raise ValueError(f"x.shape[1] = f{n_x} != u.shape[1] = {n_u}")

        return x, u, squeeze

    def _center_inputs(self, x, u, xf, uf):
        """
        Wrapper of `_reshape_inputs` that reshapes 1d array state and controls
        into 2d arrays and subtracts nominal states and controls.

        Parameters
        ----------
        x : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        u : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        xf : (n_states, 1) array
            Nominal state to subtract from `x`.
        uf : (n_controls, 1) array
            Nominal control to subtract from `u`.

        Returns
        -------
        x - xf : (n_states, n_points) array
            Centered state(s) arranged by (dimension, time). If the input was
            flat, `n_points = 1`.
        u - uf : (n_controls, n_points) array
            Centered control(s) arranged by (dimension, time). If the input was
            flat, `n_points = 1`.
        squeeze: bool
            True if either input was flat.

        Raises
        ------
        ValueError
            If cannot reshape states and controls to the correct sizes, or if
            `x.shape[1] != u.shape[1]`.
        """
        x, u, squeeze = self._reshape_inputs(x, u)
        x_err = x - np.reshape(xf, (self.n_states, 1))
        u_err = u - np.reshape(uf, (self.n_controls, 1))
        return x_err, u_err, squeeze


class ProblemParameters:
    """Utility class to store cost function and system dynamics parameters and
    allow these to be updated (during simulation, for example)."""
    def __init__(self, required=[], update_fun=None, **params):
        """
        Parameters
        ----------
        required : list or set of strings, default=[]
            Names of parameters which cannot be None.
        update_fun : callable, optional
            A function to execute whenever problem parameters are modified by
            `update`. The function must have the call signature
            `update_fun(obj, **params)` where `obj` refers to the
            `ProblemParameters` instance and `params` are parameters to be
            modified, specified as keyword arguments.
        **params : dict
            Parameters to set as initialization, as keyword arguments. Sets
            attributes of the `ProblemParameters` instance with these
            parameters, so unexpected behavior can occur if these overwrite
            class or instance attributes.
        """
        if update_fun is None:
            self._update_fun = lambda s, **p: None
        elif callable(update_fun):
            self._update_fun = update_fun
        else:
            raise TypeError('update_fun must be set with a callable')

        self._param_dict = dict()
        self.required = set(required)
        if len(params):
            self.update(**params)

    def update(self, check_required=True, **params):
        """
        Modify individual or multiple parameters using keyword arguments. This
        internally calls `self._update_fun(self, **params)`.

        Parameters
        ----------
        check_required : bool, default=True
            Ensure that all required parameters have been set (after updating).
        **params : dict
            Parameters to change, as keyword arguments. Sets attributes of the
            `ProblemParameters` instance with these parameters, so unexpected
            behavior can occur if these overwrite class or instance attributes.

        Raises
        ------
        RuntimeError
            If `check_required` is True and any of the parameters in
            `self.required` is None after updating.
        """
        self._param_dict.update(params)
        self.__dict__.update(params)

        if check_required:
            for p in self.required:
                if getattr(self, p, None) is None:
                    raise RuntimeError(f"{p} is required but has not been set")

        # Run other needed operations
        self._update_fun(self, **params)

    def as_dict(self):
        """
        Return all named parameters in the form of a dict.

        Returns
        -------
        parameter_dict : dict
            Dict containing all parameters set using `__init__` or `update`.
        """
        return self._param_dict
