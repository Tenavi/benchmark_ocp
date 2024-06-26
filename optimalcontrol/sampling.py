import numpy as np

from .utilities import check_int_input, resize_vector


class StateSampler:
    """Generic base class for algorithms to sample states."""
    def __init__(self, *args, **kwargs):
        pass

    def update(self, **kwargs):
        """Update parameters of the sampler."""
        pass

    def __call__(self, n_samples=1, **kwargs):
        """
        Generate samples of the system state.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        **kwargs : dict
            Other keyword arguments implemented by the subclass.

        Returns
        -------
        x : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x` will be a 1d array.
        """
        raise NotImplementedError


class UniformSampler(StateSampler):
    """Class which implements uniform sampling from a hypercube."""
    def __init__(self, lb, ub, xf, norm=2, seed=None):
        """
        Parameters
        ----------
        lb : {(n_states,) or (n_states, 1) array, float}
            Lower bounds for each dimension of the hypercube. If float, will be
            broadcast into an array of shape `(n_states, 1)`.
        ub : {(n_states,) or (n_states, 1) array, float}
            Upper bounds for each dimension of the hypercube. If float, will be
            broadcast into an array of shape `(n_states, 1)`.
        xf : (n_states,) or (n_states,1) array
            Nominal state within the hypercube. If `sample` is called with a
            specified `distance` argument, this distance is calculated from `xf`
            with norm specified by `norm`.
        norm : {1, 2, float, `np.inf`, (n_states, n_states) array}, default=2
            The norm (l1, l2, l-infinity, or matrix) with which to calculate
            distances from `xf`. If `norm` is an array then it must be positive
            definite. In this case it is defined as
            `distance(x) = sqrt(x.T @ self.norm @ x)`. If `norm` is a float it
            will be converted to a diagonal array and treated as an array norm.
        seed : int, optional
            Random seed for the random number generator.
        """
        self.update(lb=lb, ub=ub, xf=xf)

        self.rng = np.random.default_rng(seed)

        bad_norm = False

        try:
            if isinstance(norm, str):
                raise TypeError

            norm = np.asarray(norm)

            if norm.size == 1:
                if np.isin(norm, (1, 2, np.inf)):
                    self.norm = np.squeeze(norm)
                elif norm > 0.:
                    self.norm = np.diag(np.full(self.n_states, np.sqrt(norm)))
                else:
                    bad_norm = True
            elif norm.shape == (self.n_states, self.n_states):
                self.norm = np.linalg.cholesky(norm).T
            else:
                bad_norm = True

        except:
            bad_norm = True

        if bad_norm:
            raise ValueError('norm must be 1, 2, a float, np.inf, or a positive'
                             ' definite (n_states, n_states) array')

    def update(self, lb=None, ub=None, xf=None, seed=None):
        """
        Update parameters of the sampler.

        Parameters
        ----------
        lb : (n_states,) or (n_states,1) array, optional
            Lower bounds for each dimension of the hypercube.
        ub : (n_states,) or (n_states,1) array, optional
            Upper bounds for each dimension of the hypercube.
        xf : (n_states,) or (n_states,1) array, optional
            Nominal state within the hypercube. If `sample` is called with a
            specified `distance` argument, this distance is calculated from `xf`
            with norm specified by `norm`.
        seed : int, optional
            Random seed for the random number generator.
        """
        try:
            if xf is not None:
                self.xf = resize_vector(xf, -1)
                self.n_states = self.xf.shape[0]
            if lb is not None:
                self.lb = resize_vector(lb, self.n_states)
            if ub is not None:
                self.ub = resize_vector(ub, self.n_states)
        except:
            raise ValueError('lb, ub, and xf must have compatible shapes')

        if np.any(self.xf > self.ub) or np.any(self.lb > self.xf):
            raise ValueError('Must have lb <= xf <= ub')

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, n_samples=1, distance=None):
        """
        Generate samples of the system state uniformly in a hypercube with lower
        and upper bounds specified by `self.lb` and `self.ub`, respectively.
        Optionally, samples may instead have a specified distance from
        equilibrium.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : float, optional
            Desired distance of samples from `self.xf`. The distance metric is
            determined by `self.norm`. Negative distance is equivalent to
            positive distance. Note that depending on how `distance` is
            specified, samples may be outside the hypercube.

        Returns
        -------
        x : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x` will be a 1d array.
        """
        n_samples = check_int_input(n_samples, 'n_samples', low=1)

        x = self.rng.uniform(low=self.lb, high=self.ub,
                             size=(self.n_states, n_samples))

        if distance is not None:
            x -= self.xf
            if np.ndim(self.norm) < 2:
                x_norm = np.linalg.norm(x, self.norm, axis=0)
            else:
                x_norm = np.einsum('ij,js->is', self.norm, x)
                x_norm = np.einsum('is,is->s', x_norm, x_norm)
                x_norm = np.sqrt(x_norm)
            non_zero_norm = x_norm > 0.
            x[:, non_zero_norm] *= distance / x_norm[non_zero_norm]
            x += self.xf

        if n_samples == 1:
            return x[:, 0]
        return x
