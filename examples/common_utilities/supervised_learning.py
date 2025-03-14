import time

import numpy as np
from sklearn import linear_model as sk_linear_models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (RobustScaler, PolynomialFeatures,
                                   FunctionTransformer)

from optimalcontrol import controls, utilities, open_loop


class SupervisedController(controls.Controller):
    """
    Template class for how one might implement a controller trained by
    supervised learning.

    To do this, we generate a dataset of state-control pairs
    (`x_data`, `u_data`), and train a regression model of the mapping from
    `x_data` to `u_data`.
    """
    def __init__(self, u_lb=None, u_ub=None, **options):
        """
        Parameters
        ----------
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            Upper control saturation bounds.
        **options : dict
            Keyword arguments to pass to the regression model.
        """
        self.u_lb, self.u_ub = u_lb, u_ub

        self._x_scaler = RobustScaler()
        self._u_scaler = RobustScaler()

        self._options = options
        self._regressor = None

        self.train_time = 0.

    @property
    def n_states(self):
        return self._x_scaler.n_features_in_

    @property
    def n_controls(self):
        return self._u_scaler.n_features_in_

    def train(self, x_data, u_data):
        """
        Parameters
        ----------
        x_data : (n_states, n_data) array
            A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        u_data : (n_controls, n_data) array
            The optimal feedback controls evaluated at the states `x_data`.
        """
        start_time = time.time()

        x_data = np.transpose(np.atleast_2d(x_data))
        u_data = np.transpose(np.atleast_2d(u_data))

        # Scale the input and output data based on the interquartile range
        self._x_scaler.fit(x_data)
        self._u_scaler.fit(u_data)

        self._regressor = self._fit_regressor(
            self._x_scaler.transform(x_data),
            np.squeeze(self._u_scaler.transform(u_data)),
            **self._options)

        if self.u_lb is not None:
            self.u_lb = utilities.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = utilities.resize_vector(self.u_ub, self.n_controls)

        self.train_time = time.time() - start_time
        if self._options.get('verbose', True):
            print(f"Training time: {self.train_time:.2f} seconds")

    def _fit_regressor(self, x_scaled, u_scaled, **options):
        """
        Fit the underlying regression model to data.

        Parameters
        ----------
        x_scaled : (n_data, n_states) array
            A set of system states. Assumed to be scaled to a numerical range
            suitable for optimization.
        u_scaled : (n_data, n_controls) or (n_data,) array
            The optimal feedback controls evaluated at the states `x_data`.
            Assumed to be scaled to a numerical range suitable for optimization.
        **options : dict
            Keyword arguments to pass to the regression model.

        Returns
        -------
        regressor : object
            The trained regression map from `x_scaled` to `u_scaled`.
        """
        raise NotImplementedError

    def __call__(self, x):
        x_T = np.reshape(x, (self.n_states, -1)).T
        x_T = self._x_scaler.transform(x_T)

        u = self._regressor.predict(x_T).reshape(-1, self.n_controls)
        u = self._u_scaler.inverse_transform(u)
        u = utilities.saturate(u.T, self.u_lb, self.u_ub)

        if np.ndim(x) < 2:
            return u[:, 0]

        return u


class NeuralNetworkController(SupervisedController):
    """
    A simple example of how one might implement a neural network control law
    trained by supervised learning.

    To do this, we generate a dataset of state-control pairs
    (`x_data`, `u_data`), and optionally for time-dependent problems, associated
    time values `t_data`, and the neural network learns the mapping from
    `x_data` (and `t_data`) to `u_data`. The neural network is implemented with
    `sklearn.neural_network.MLPRegressor`.
    """
    def _fit_regressor(self, x_scaled, u_scaled, **options):
        nn = MLPRegressor(**options)
        return nn.fit(x_scaled, u_scaled)


class KNeighborsController(SupervisedController):
    """
    A simple example of how one might implement a K-nearest neighbors (KNN)
    control law trained by supervised learning.

    To do this, we generate a dataset of state-control pairs
    (`x_data`, `u_data`), and optionally for time-dependent problems, associated
    time values `t_data`, and the KNN regressor learns the mapping from `x_data`
    (and `t_data`) to `u_data`. KNN regression implemented with
    `sklearn.neighbors.KNeighborsRegressor`.
    """
    def _fit_regressor(self, x_scaled, u_scaled, **options):
        knn = KNeighborsRegressor(**options)
        return knn.fit(x_scaled, u_scaled)


class PolynomialController(SupervisedController):
    """
    A simple example of how one might implement a polynomial control law trained
    by supervised learning.

    To do this, we generate a dataset of state-control pairs
    (`x_data`, `u_data`), and optionally for time-dependent problems, associated
    time values `t_data`, and the regressor learns a polynomial mapping from
    `x_data` (and `t_data`) to `u_data`. The polynomial is implemented with
    `sklearn.preprocessing.PolynomialFeatures` and a choice of model from
    `sklearn.linear_model`, controlled by the `linear_model` keyword
    (default='BayesianRidge'). Note that if `n_controls > 1` and a cross
    validation-based `linear_model` is used, this is not natively supported in
    `sklearn` so the regressor will be wrapped with
    `sklearn.multioutput.MultiOutputRegressor`.
    """
    def _fit_regressor(self, x_scaled, u_scaled, degree=1,
                       linear_model='BayesianRidge', **options):
        regressor = getattr(sk_linear_models, linear_model)(**options)

        regressor = Pipeline([('kernel', PolynomialFeatures(degree=degree)),
                              ('regressor', regressor)])

        try:
            return regressor.fit(x_scaled, u_scaled)
        except ValueError:
            return MultiOutputRegressor(regressor).fit(x_scaled, u_scaled)


class RandomFourierController(SupervisedController):
    """
    A simple example of how one might implement a random Fourier feature control
    law trained by supervised learning.

    To do this, we generate a dataset of state-control pairs
    (`x_data`, `u_data`), and optionally for time-dependent problems, associated
    time values `t_data`, and the regressor learns a random Fourier feature
    mapping (see ref. [1]) from `x_data` (and `t_data`) to `u_data`.

    A random Fourier feature mapping is formulated as follows. Let `x` and `u`
    be scaled inputs and outputs of dimensions `n_states` and `n_controls`,
    respectively. We choose `n_features`, `sigma`, `random_state`, and
    `linear_model` parameters (see below). We sample a matrix of frequencies,
    `W`, from a normal distribution, and a vector of biases, `B`, uniformly:
    ```
    rng = np.random.default_rng(random_state)
    W = rng.normal(scale=sigma, size=(n_features, n_states))
    B = rng.uniform(low=0, high=2 * np.pi, size=(n_features, 1))
    ```
    These frequencies and biases are instantiated and fixed when the model is
    first trained. We then learn a linear model of the form
    ```
    z = cos(W @ x + B)
    u = K @ z + C
    ```
    where `K` and `C` are `(n_controls, n_features)` and `(n_controls, 1)`
    matrices, respectively. The linear regression step is implemented with
    `sklearn.linear_model`.

    ##### References
    1. A. Rahimi and B. Recht, Random features for large-scale kernel machines,
        in Advances in Neural Information Processing Systems, 2007, pp.
        1177-1184.

    Parameters
    ----------
    u_lb : (`n_controls`, 1) array, optional
        Lower control saturation bounds.
    u_ub : (`n_controls`, 1) array, optional
        Upper control saturation bounds.
    n_features : int, default=`10 * n_states * n_controls`
        Number of Fourier features (i.e., random frequencies).
    sigma : {float, (`n_states`,) array}, default=0.5
        Standard deviation of the random frequencies for each input. Note that
        inputs are scaled using `sklearn.preprocessing.RobustScaler`.
    random_state : int, optional
        Seed for the random number generator.
    linear_model : str, default='BayesianRidge'
        Linear regression method to use. Can be anything implemented in
        `sklearn.linear_model`. Note that if `n_u > 1` and a cross
        validation-based `linear_model` is used, this is not natively supported
        in `sklearn` so the regressor will be wrapped with
        `sklearn.multioutput.MultiOutputRegressor`.
    **options : dict
        Keyword arguments to pass to the linear regression model.
    """
    def __init__(self, u_lb=None, u_ub=None, n_features=None, sigma=0.5,
                 random_state=None, linear_model='BayesianRidge', **options):
        super().__init__(u_lb=u_lb, u_ub=u_ub, n_features=n_features,
                         sigma=sigma, linear_model=linear_model,
                         random_state=random_state, **options)
        self.W = None
        """`(n_features, n_states)` array. Matrix of random frequencies."""
        self.B = None
        """`(n_features, 1)` array. Vector of random frequency biases."""
        self._rng = np.random.default_rng(random_state)

    def _fit_regressor(self, x_scaled, u_scaled, n_features=None, sigma=0.5,
                       random_state=None, linear_model='BayesianRidge',
                       **options):
        # Initialize frequencies and bias
        if n_features is None:
            n_features = 10 * self.n_states * self.n_controls

        w_shape = (n_features, self.n_states)
        if self.W is None or self.W.shape != w_shape:
            self.W = self._rng.normal(scale=np.reshape(sigma, -1), size=w_shape)

        b_shape = (n_features, 1)
        if self.B is None or self.B.shape != b_shape:
            self.B = self._rng.uniform(high=2. * np.pi, size=b_shape)

        kernel = FunctionTransformer(self._map_to_fourier, validate=True)

        regressor = getattr(sk_linear_models, linear_model)
        try:
            regressor = regressor(random_state=random_state, **options)
        # In case the linear_model doesn't take a random_state parameter
        except TypeError:
            regressor = regressor(**options)

        regressor = Pipeline([('kernel', kernel), ('regressor', regressor)])

        try:
            return regressor.fit(x_scaled, u_scaled)
        except ValueError:
            return MultiOutputRegressor(regressor).fit(x_scaled, u_scaled)

    def _map_to_fourier(self, x):
        return np.cos(x @ self.W.T + self.B.T)


class SimpleQRnet(SupervisedController):
    """
    Example of the basic u-QRnet method from ref. [1], which combines a
    regression model (such as a NN) with an LQR controller.

    The regressor is trained using supervised learning. To do this, we generate
    a dataset of state-control pairs (`x_data`, `u_data`), and train a
    regression model of the mapping from `x_data` to `u_data - u_lqr`, where
    `u_lqr` is the output of an LQR controller for the linearized problem. In
    the closed-loop, the control is given by `u_model + u_lqr - u_model(xf)`,
    where `u_model` is the regression output and `xf = lqr.xf` is the
    linearization point.

    Note that this basic method promotes but does not guarantee local stability,
    unlike the more advanced methods in ref. [2]. Furthermore, in this
    simplified implementation, the `u_model(xf)` term is not included during
    training--it is added afterward. This should not make a large impact, but
    including this term in training would likely improve accuracy.

    ##### References
    1. T. Nakamura-Zimmerer, Q. Gong, and W. Kang, Neural Network Optimal
        Feedback Control with Enhanced Closed Loop Stability, in American
        Control Conference, 2022, pp. 2373-2378.
        https://doi.org/10.23919/ACC53348.2022.9867619
    2. T. Nakamura-Zimmerer, Q. Gong, and W. Kang, Neural Network Optimal
        Feedback Control with Guaranteed Local Stability, IEEE Open Journal of
        Control Systems, 1 (2022), pp. 210-222.
        https://doi.org/10.1109/OJCSYS.2022.3205863
    """
    def __init__(self, lqr, controller):
        """
        Parameters
        ----------
        lqr : `LinearQuadraticRegulator`
            Instance of `LinearQuadraticRegulator` for the linearized optimal
            control problem.
        controller : `SupervisedController`
            Instance of a `SupervisedController` subclass which is used to model
            nonlinear parts of the optimal control.
        """
        super().__init__(u_lb=lqr.u_lb, u_ub=lqr.u_ub)

        self.wrapped_controller = controller
        self._lqr = lqr

        self.xf = lqr.xf
        self._wrapped_uf = None

        self._x_scaler = controller._x_scaler
        self._u_scaler = controller._u_scaler

    def __str__(self):
        return f"{str(self.wrapped_controller).replace('Controller', '')}+LQR"

    def train(self, x_data, u_data):
        self.wrapped_controller.train(x_data, u_data - self._lqr(x_data))
        self._wrapped_uf = self.wrapped_controller(self.xf)
        self.train_time = self.wrapped_controller.train_time

    def __call__(self, x):
        u_lqr = self._lqr(x)

        u_model = self.wrapped_controller(x)

        u = u_model + u_lqr

        if u.ndim < 2:
            u -= self._wrapped_uf.reshape(-1)
        else:
            u -= self._wrapped_uf

        return utilities.saturate(u, self.u_lb, self.u_ub)


class QuaternionControlWrapper(SupervisedController):
    """
    Wrapper of another `SupervisedController` which always treats a scalar
    quaternion state as positive.
    """
    def __init__(self, q0_state, controller):
        """
        Parameters
        ----------
        q0_state : int
            Index of the state which contains the scalar quaternion which should
            be transformed to be positive.
        controller : `SupervisedController`
            Instance of a `SupervisedController` subclass which is used to model
            the optimal control.
        """
        super().__init__()

        self.wrapped_controller = controller
        self._q0_state = int(q0_state)

        self._x_scaler = self.wrapped_controller._x_scaler
        self._u_scaler = self.wrapped_controller._u_scaler

    def __str__(self):
        return str(self.wrapped_controller)

    def _make_positive_q0(self, x_in):
        x_out = np.copy(x_in)
        x_out[self._q0_state] = np.abs(x_out[self._q0_state])
        return x_out

    def train(self, x_data, u_data):
        self.wrapped_controller.train(self._make_positive_q0(x_data), u_data)
        self.u_lb = self.wrapped_controller.u_lb
        self.u_ub = self.wrapped_controller.u_ub
        self.train_time = self.wrapped_controller.train_time

    def __call__(self, x):
        return self.wrapped_controller(self._make_positive_q0(x))


def generate_data(ocp, guesses, verbose=0, **kwargs):
    """
    Given an existing open loop data set, resolve the open loop OCP using the
    previously generated data as initial guesses. Used when refining solutions
    with an indirect method or higher tolerances.

    Parameters
    ----------
    ocp : `OptimalControlProblem`
        An instance of an `OptimalControlProblem` subclass implementing
        `bvp_dynamics` and `hamiltonian_minimizer` methods.
    guesses : length n_problems list of dicts,
        Initial guesses for each open-loop OCP. Each element of `guesses` should
        be a dict or DataFrame with keys

            * t : (n_points,) array
                Time points.
            * x : (`ocp.n_states`, n_points) array
                Guess for system states at times `t`.
            * u : (`ocp.n_controls`, n_points) array, optional
                Guess for optimal controls at times `t`. Required if
                `method=='direct'`.
            * p : (`ocp.n_states`, n_points) array, optional
                Guess for costates at times `t`. Required if
                `method=='indirect'` (default).
            * v : (n_points,) array, optional
                Guess for value function at times `t`.
    verbose : {0, 1, 2}, default=0
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    **kwargs : dict
        Keyword arguments to pass to the solver. For fixed final time problems,
        see `optimalcontrol.open_loop.solve_fixed_time`. For infinite horizon
        problem, see `optimalcontrol.open_loop.solve_infinite_horizon`.

    Returns
    -------
    data : (n_problems,) object array of dicts
        Solutions or attempted solutions of the open loop OCP based on
        `guesses`. Each element is a dict with the same keys and values as
        `guesses`. If `status[i]==0`, then `data[i]` is considered an acceptable
        solution, otherwise `data[i]` contains the solver's best attempt at a
        solution upon failure (which may simply be the original `guesses[i]`).
    status : (n_problems,) integer array
        `status[i]` contains an int describing if an acceptable solution based
        on `guesses[i]` was found. In particular, if `status[i]==0` then
        `data[i]` was deemed acceptable.
    messages : (n_problems,) string array
        `messages[i]` contains a human-readable message describing `status[i]`.
    """
    if np.isinf(ocp.final_time):
        sol_fun = open_loop.solve_infinite_horizon
    else:
        sol_fun = open_loop.solve_fixed_time

    if type(guesses) not in (list, np.ndarray):
        guesses = [guesses]

    data = []
    status = np.zeros(len(guesses), dtype=int)
    messages = []

    n_succeed = 0

    sol_time = 0.
    fail_time = 0.

    print("\nSolving open loop optimal control problems...")
    w = str(len('attempted') + 2)
    row = '{:^' + w + '}|{:^' + w + '}|{:^' + w + '}|{:^' + w + '}'
    h1 = row.format('solved', 'attempted', 'desired', 'elapsed')
    headers = ('\n' + h1, len(h1) * '-')

    for header in headers:
        print(header)

    for i, guess in enumerate(guesses):
        t, x, u, p, v = utilities.unpack_dataframe(guess)

        start_time = time.time()

        sol = sol_fun(ocp, t, x, u=u, p=p, v=v, verbose=verbose, **kwargs)

        end_time = time.time()

        status[i] = sol.status
        messages.append(sol.message)

        if status[i] == 0:
            sol_time += end_time - start_time
            n_succeed += 1
        else:
            fail_time += end_time - start_time

        data.append({'t': sol.t, 'x': sol.x, 'u': sol.u, 'p': sol.p, 'v': sol.v,
                     'L': ocp.running_cost(sol.x, sol.u)})

        if verbose:
            for header in headers:
                print(header)

        overwrite = not verbose and i + 1 < len(guesses)

        total_time = sol_time + fail_time
        total_time = ("{:" + str(int(w) - 6) + ".0f} s").format(total_time)
        print(row.format(n_succeed, i + 1, len(guesses), total_time),
              end='\r' if overwrite else None)

    print("\nTotal solution time:")
    print(f"    Successes: {sol_time:.1f} seconds")
    print(f"    Failures : {fail_time:.1f} seconds")

    if n_succeed < len(guesses):
        print("\nFailed initial conditions:")
        for i, stat in enumerate(status):
            if stat != 0:
                print(f"i={i:d} : status = {stat:d} : {messages[i]:s}")

    data = np.asarray(data, dtype=object)
    messages = np.asarray(messages, dtype=str)

    return data, status, messages
