import pytest

import numpy as np

from optimalcontrol.utilities import StateSampler, UniformSampler

rng = np.random.default_rng()

def test_UniformSampler_init():
    n_states = 4
    ub = np.arange(1.,1. + n_states)
    lb = -ub
    xf = rng.uniform(low=-1., high=1., size=n_states)

    # Make sure parameters are initialized correctly and with correct shapes
    sampler = UniformSampler(lb, ub, xf)
    for key, var in zip(('lb', 'ub', 'xf'), (lb, ub, xf)):
        assert sampler.__dict__[key].ndim == 2
        assert sampler.__dict__[key].shape[0] == n_states
        assert sampler.__dict__[key].shape[1] == 1
        assert np.allclose(sampler.__dict__[key].flatten(), var)

    sampler = UniformSampler(lb.reshape(-1,1), list(ub), xf, norm=np.array(1))
    sampler = UniformSampler(lb, ub.reshape(-1,1), list(xf), norm=np.array([2]))
    sampler = UniformSampler(list(lb), ub, xf.reshape(-1,1), norm=np.array([[1]]))

def test_UniformSampler_bad_init():
    n_states = 4
    ub = np.arange(1.,1. + n_states)
    lb = -ub
    xf = rng.uniform(low=-1., high=1., size=n_states)

    with pytest.raises(ValueError):
        sampler = UniformSampler(lb, ub, xf[:-1])

    with pytest.raises(ValueError):
        sampler = UniformSampler(lb, ub[:-1], xf)

    with pytest.raises(ValueError):
        sampler = UniformSampler(lb[:-1], ub, xf)

    with pytest.raises(ValueError):
        _lb = np.copy(lb)
        _lb[0] = ub[0] + 1.
        sampler = UniformSampler(_lb, ub, 0.*xf)

    with pytest.raises(ValueError):
        sampler = UniformSampler(lb, ub, xf + ub.max())

    for bad_norm in [0,1.5,'xyz']:
        with pytest.raises(ValueError):
            sampler = UniformSampler(lb, ub, xf, norm=bad_norm)

@pytest.mark.parametrize('norm', [1,2])
@pytest.mark.parametrize('distance', [None,1.5])
def test_UniformSampler_sample(norm, distance):
    n_states, seed = 4, 123
    ub = np.arange(1.,1. + n_states).reshape(n_states,1)
    lb = -ub
    xf = rng.uniform(low=-1., high=1., size=(n_states,1))
    sampler = UniformSampler(lb, ub, xf, norm=norm, seed=seed)

    with pytest.raises(Exception):
        problem.sample_initial_conditions(n_samples=0)

    for n_samples in range(1,4):
        sampler.update(seed=seed)
        x0 = sampler(n_samples=n_samples, distance=distance)

        # Check for correct sizes
        if n_samples == 1:
            assert x0.ndim == 1
            x0 = x0.reshape(-1,1)
        else:
            assert x0.ndim == 2
            assert x0.shape[1] == n_samples
        assert x0.shape[0] == n_states

        # Check that samples are consistent if the seed is reset
        sampler.update(seed=seed)
        x1 = sampler(n_samples=n_samples, distance=distance)
        assert np.allclose(x0, x1.reshape(x0.shape))

        # Check that either bounds are satisfied or desired distance reached
        if distance is None:
            assert np.all(x0 <= ub)
            assert np.all(lb <= x0)
        else:
            np.allclose(distance, np.linalg.norm(x0 - xf, ord=norm, axis=0))
