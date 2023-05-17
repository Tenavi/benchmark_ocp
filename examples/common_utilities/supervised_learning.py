import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model as sk_linear_models
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score

from optimalcontrol import controls, utilities


class SupervisedController(controls.Controller):
    """
    A simple example of how one might implement a controller trained by
    supervised learning. To do this, we generate a dataset of state-control
    pairs (`x_data`, `u_data`), and optionally for time-dependent problems,
    associated time values `t_data`, and train a regression model of the mapping
    from `x_data` (and `t_data`) to `u_data`.
    """
    def __init__(self, x_data, u_data, t_data=None, u_lb=None, u_ub=None,
                 **options):
        """
        Parameters
        ----------
        x_data : (n_states, n_data) array
            A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        u_data : (n_controls, n_data) array
            The optimal feedback controls evaluated at the states `x_data`.
        t_data : (n_data,) array, optional
            For time-dependent problems, the time values at which the pairs
            (`x_data`, `u_data`) are obtained.
        u_lb : (n_controls, 1) array, optional
            Lower control saturation bounds.
        u_ub : (n_controls, 1) array, optional
            Upper control saturation bounds.
        **options : dict
            Keyword arguments to pass to the regression model.
        """
        start_time = time.time()

        x_data = np.transpose(np.atleast_2d(x_data))
        if t_data is not None:
            raise NotImplementedError("Time-dependent problems are not yet "
                                      "implemented")
        u_data = np.transpose(np.atleast_2d(u_data))

        # Scale the input and output data based on the interquartile range
        self._x_scaler = RobustScaler().fit(x_data)
        self._u_scaler = RobustScaler().fit(u_data)

        if t_data is None:
            self.n_states = self._x_scaler.n_features_in_
        else:
            self.n_states = self._x_scaler.n_features_in_ - 1

        self.n_controls = self._u_scaler.n_features_in_

        self._regressor = self._fit_regressor(
            self._x_scaler.transform(x_data),
            np.squeeze(self._u_scaler.transform(u_data)), **options)

        self.u_lb, self.u_ub = u_lb, u_ub

        if self.u_lb is not None:
            self.u_lb = utilities.resize_vector(self.u_lb, self.n_controls)
        if self.u_ub is not None:
            self.u_ub = utilities.resize_vector(self.u_ub, self.n_controls)

        self._train_time = time.time() - start_time
        if options.get('verbose', True):
            print(f"\nTraining time: {self._train_time:.2f} seconds")

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
            return u.flatten()

        return u

    def r2_score(self, x_data, u_data, multioutput='uniform_average'):
        r"""
        Return the coefficient of determination of the control prediction in the
        physical (unscaled) domain.

        The coefficient of determination, $R^2$, is defined as
        `r2 = 1 - residual / total`, where
        `residual = ((u_data - u_pred)**2).sum()` with `u_pred = self(x_data)`,
        and `total = ((u_data - u_data.mean()) ** 2).sum()`. The best possible
        score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of `u_data`, disregarding the input features, would get an $R^2$
        score of 0.0.
        
        Parameters
        ----------
        x_data : (n_states, n_data) array
            A set of system states (obtained by solving a set of open-loop
            optimal control problems).
        u_data : (n_controls, n_data) array
            The optimal feedback controls evaluated at the states `x_data`.
        multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
                (n_controls,) array, or None, default='uniform_average'

            Defines aggregating of multiple output scores. An array value
            defines weights used to average scores, and None reverts to the
            default 'uniform_average'.

                * 'raw_values' : Returns a full set of scores for each control.

                * 'uniform_average' :
                    Scores of all control dimensions are averaged with uniform
                    weight.

                * 'variance_weighted' :
                    Scores of all control dimensions are averaged, weighted by
                    the variances of each individual control.

        Returns
        -------
        r2 : float or (n_controls,) array
            The $R^2$ score, or array of scores if `multioutput=='raw_values'`.
        """
        u_pred = self(x_data)
        u_data = np.reshape(u_data, u_pred.shape)
        return r2_score(u_data.T, u_pred.T, multioutput=multioutput)


class NeuralNetworkController(SupervisedController):
    """
    A simple example of how one might implement a neural network control law
    trained by supervised learning. To do this, we generate a dataset of
    state-control pairs (`x_data`, `u_data`), and optionally for time-dependent
    problems, associated time values `t_data`, and the neural network learns the
    mapping from `x_data` (and `t_data`) to `u_data`. The neural network is
    implemented with `sklearn.neural_network.MLPRegressor`.
    """
    def _fit_regressor(self, x_scaled, u_scaled, **options):
        nn = MLPRegressor(**options)
        return nn.fit(x_scaled, u_scaled)


class PolynomialController(SupervisedController):
    """
    A simple example of how one might implement a polynomial control law trained
    by supervised learning. To do this, we generate a dataset of state-control
    pairs (`x_data`, `u_data`), and optionally for time-dependent problems,
    associated time values `t_data`, and the polynomial regressor learns the
    mapping from `x_data` (and `t_data`) to `u_data`. The polynomial is
    implemented with `sklearn.preprocessing.PolynomialFeatures` and a choice of
    model from `sklearn.linear_model`, controlled by the `linear_model` keyword
    (default='Ridge'). Note that if `n_controls > 1` and a cross alidation-based
    `linear_model` is used, this is not natively supported in `sklearn` so the
    regressor will be wrapped with `sklearn.multioutput.MultiOutputRegressor`.
    """
    def _fit_regressor(self, x_scaled, u_scaled, degree=1, linear_model='Ridge',
                       **options):

        regressor = getattr(sk_linear_models, linear_model)(**options)

        if self.n_controls > 1 and linear_model[-2:] == 'CV':
            regressor = MultiOutputRegressor(regressor)

        regressor = Pipeline([('kernel', PolynomialFeatures(degree=degree)),
                              ('regressor', regressor)])
        return regressor.fit(x_scaled, u_scaled)
