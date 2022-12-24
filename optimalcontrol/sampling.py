import numpy as np

class StateSampler:
    '''Generic base class for algorithms to sample states.'''
    def __init__(self, *args, **kwargs):
        pass

    def update(self, **kwargs):
        '''Update parameters of the sampler.'''
        pass

    def __call__(self, n_samples=1, **kwargs):
        '''
        Generate samples of the system state.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        kwargs
            Other keyword arguments implemented by the subclass.

        Returns
        -------
        x : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x` will be a one-dimensional array.
        '''
        raise NotImplementedError

class UniformSampler(StateSampler):
    '''Class which implements uniform sampling from a hypercube.'''
    def __init__(self, lb, ub, xf, norm=2, seed=None):
        '''
        Parameters
        ----------
        lb : (n_states,) or (n_states,1) array
            Lower bounds for each dimension of the hypercube.
        ub : (n_states,) or (n_states,1) array
            Upper bounds for each dimension of the hypercube.
        xf : (n_states,) or (n_states,1) array
            Nominal state within the hypercube. If `sample` is called with a
            specified `distance` argument, this distance is calculated from `xf`
            with norm specified by `norm`.
        norm : {1, 2, (n_states, n_states) array}, default=2
            The norm (l1, l2, or matrix) with which to calculate distances from
            `xf`. If `norm` is an array then it must be positive semi-definite.
            In this case it is defined as
                `||x|| = sqrt(x.T @ self.Q @ x)`
            where `Q` is the given matrix.
        seed : int, optional
            Random seed for the random number generator.
        '''
        self.update(lb=lb, ub=ub, xf=xf)

        if np.size(norm) == 1 and norm in [1,2]:
            self.norm = int(norm)
        elif np.shape(norm) == (self.n_states, self.n_states):
            try:
                self.norm = np.linalg.cholesky(norm).T
            except:
                raise ValueError('If norm is an array it must be positive definite.')
        else:
            raise ValueError('norm must be 1, 2, or an (n_states, n_states) array.')

        self.rng = np.random.default_rng(seed)

    def update(self, lb=None, ub=None, xf=None, seed=None):
        '''
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
        '''
        if lb is not None:
            self.lb = np.reshape(lb, (-1,1))
        if ub is not None:
            self.ub = np.reshape(ub, (-1,1))
        if xf is not None:
            self.xf = np.reshape(xf, (-1,1))
            self.n_states = self.xf.shape[0]

        if self.n_states != self.ub.shape[0] or self.n_states != self.lb.shape[0]:
            raise ValueError('lb, ub, and xf must have compatible shapes.')

        if not np.all(self.xf <= self.ub) or not np.all(self.lb <= self.xf):
            raise ValueError('Must have lb <= xf <= ub.')

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, n_samples=1, distance=None):
        '''
        Generate samples of the system state uniformly in a hypercube with lower
        and upper bound specified by `self.lb` and `self.ub`, respectively.

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
            sample. If `n_samples=1` then `x` will be a one-dimensional array.
        '''
        if not n_samples:
            raise ValueError('n_samples must be a positive int.')

        x = self.rng.uniform(
            low=self.lb, high=self.ub, size=(self.n_states, n_samples)
        )

        if distance is not None:
            x -= self.xf
            if isinstance(self.norm, int):
                x_norm = distance / np.linalg.norm(x, self.norm, axis=0)
            else:
                x_norm = np.einsum('ij,js->is', self.norm, x)
                x_norm = np.einsum('is,is->s', x_norm, x_norm)
                x_norm = distance / np.sqrt(x_norm)
            x *= x_norm
            x += self.xf

        if n_samples == 1:
            return x.flatten()
        return x