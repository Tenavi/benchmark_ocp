import pytest

import numpy as np

from optimalcontrol import utilities


rng = np.random.default_rng()


@pytest.mark.parametrize('n', [0., 1.5, np.array([[5], [6]]), [7, 8], 'n'])
def test_check_int_input_bad_type(n):
    """Make sure `check_int_input` catches a range of bad input types."""
    with pytest.raises(TypeError):
        utilities.check_int_input(n, 'n')


@pytest.mark.parametrize('low', [-7., np.pi, np.array([[1, 2]]), [3, 4], 'm'])
def test_check_int_input_bad_low(low):
    """Make sure `check_int_input` catches a range of bad `low` input types."""
    with pytest.raises(TypeError):
        utilities.check_int_input(rng.choice(100), 'n', low=low)


@pytest.mark.parametrize('shape', [(), (1,), (1, 1)])
def test_check_int_input(shape):
    """Make sure `check_int_input` works with a variety of acceptable inputs."""
    n = rng.choice(100, size=shape) - 50
    _n = int(np.squeeze(n))
    assert utilities.check_int_input(n.tolist(), 'n') == _n
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        assert utilities.check_int_input(n.astype(dtype), 'n') == _n


@pytest.mark.parametrize('low', [0, 1, -5, 10])
def test_check_int_input_with_low(low):
    """Make sure `check_int_input` works with minimum inputs and throws an error
    if the desired minimum is not found."""
    assert utilities.check_int_input(low, 'n', low=low) == low
    assert utilities.check_int_input(low + 1, 'n', low=low) == low + 1
    with pytest.raises(ValueError):
        utilities.check_int_input(low - 1, 'n', low=low)


@pytest.mark.parametrize('n_rows', [1, 2, 3, -1])
def test_resize_vector(n_rows):
    if n_rows > 0:
        x = rng.normal(size=n_rows)
    else:
        x = rng.normal(size=17)
    y_expect = x.reshape(-1, 1)

    y = utilities.resize_vector(x, n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.reshape(-1, 1), n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.reshape(1, -1), n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.tolist(), n_rows)
    np.testing.assert_allclose(y, y_expect)

    if n_rows > 2:
        with pytest.raises(ValueError):
            _ = utilities.resize_vector(x[:-1], n_rows)


@pytest.mark.parametrize('n_rows', [2, 3])
def test_resize_vector_float_in(n_rows):
    x = np.full(n_rows, rng.normal())
    y_expect = x.reshape(-1, 1)

    y = utilities.resize_vector(x[0], n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x[:1], n_rows)
    np.testing.assert_allclose(y, y_expect)


@pytest.mark.parametrize('n_controls', (1, 2))
@pytest.mark.parametrize('lb', (None, -rng.uniform()))
@pytest.mark.parametrize('ub', (None, rng.uniform()))
def test_saturate(n_controls, lb, ub):
    """Test that the saturation function works given different input shapes."""
    n_points = 30
    u = rng.uniform(low=-1.5, high=1.5, size=(n_controls, n_points))

    # Brute force construction of saturated controls
    u_sat_ref = u.copy()
    for d in range(n_controls):
        for k in range(n_points):
            if lb is not None and u_sat_ref[d, k] < lb:
                u_sat_ref[d, k] = lb
            elif ub is not None and ub < u_sat_ref[d, k]:
                u_sat_ref[d, k] = ub

    # Test float bounds at single points
    for k in range(n_points):
        for d in range(n_controls):
            u_sat = utilities.saturate(u[d, k], lb, ub)
            assert u_sat.shape == ()
            np.testing.assert_array_equal(u_sat, u_sat_ref[d, k])

        u_sat = utilities.saturate(u[:, k], lb, ub)
        np.testing.assert_array_equal(u_sat, u_sat_ref[:, k])

    # Test float bounds at all points
    # one-dimensional control
    for d in range(n_controls):
        u_sat = utilities.saturate(u[d], lb, ub)
        np.testing.assert_array_equal(u_sat, u_sat_ref[d])

    # two-dimensional control
    u_sat = utilities.saturate(u, lb, ub)
    np.testing.assert_array_equal(u_sat, u_sat_ref)

    # Test vector bounds and (n_controls, 1) matrix bounds
    if lb is not None or ub is not None:
        # Construct vector and matrix bounds
        lb_vec = np.full(n_controls, lb) if lb is not None else None
        ub_vec = np.full(n_controls, ub) if ub is not None else None

        lb_mat = np.full((n_controls, 1), lb) if lb is not None else None
        ub_mat = np.full((n_controls, 1), ub) if ub is not None else None

        # Test at single points
        for k in range(n_points):
            u_sat = utilities.saturate(u[:, k], lb_vec, ub_vec)
            np.testing.assert_array_equal(u_sat, u_sat_ref[:, k])

            u_sat = utilities.saturate(u[:, k], lb_mat, ub_mat)
            np.testing.assert_array_equal(u_sat, u_sat_ref[:, k])

            u_sat = utilities.saturate(u[:, k:k+1], lb_vec, ub_vec)
            np.testing.assert_array_equal(u_sat, u_sat_ref[:, k:k+1])

            u_sat = utilities.saturate(u[:, k:k+1], lb_mat, ub_mat)
            np.testing.assert_array_equal(u_sat, u_sat_ref[:, k:k+1])

        # Test at all points
        u_sat = utilities.saturate(u, lb_vec, ub_vec)
        np.testing.assert_array_equal(u_sat, u_sat_ref)

        u_sat = utilities.saturate(u, lb_mat, ub_mat)
        np.testing.assert_array_equal(u_sat, u_sat_ref)


@pytest.mark.parametrize('n_controls', (1, 2))
@pytest.mark.parametrize('lb', (None, -rng.uniform()))
@pytest.mark.parametrize('ub', (None, rng.uniform()))
def test_find_saturated(n_controls, lb, ub):
    """Test that the function for finding saturated controls works given
    different input shapes."""
    n_points = 30
    u_unsat = rng.uniform(low=-1.5, high=1.5, size=(n_controls, n_points))
    u_sat = utilities.saturate(u_unsat, lb, ub)
    sat_idx_ref = u_unsat != u_sat

    # Saturation should be detected for both saturated and unsaturated controls
    for u in (u_unsat, u_sat):
        # Test float bounds at single points
        for k in range(n_points):
            for d in range(n_controls):
                sat_idx = utilities.find_saturated(u[d, k], lb, ub)
                assert sat_idx.shape == ()
                assert sat_idx == sat_idx_ref[d, k]

            sat_idx = utilities.find_saturated(u[:, k], lb, ub)
            np.testing.assert_array_equal(sat_idx, sat_idx_ref[:, k])

        # Test float bounds at all points
        # one-dimensional control
        for d in range(n_controls):
            sat_idx = utilities.find_saturated(u[d], lb, ub)
            np.testing.assert_array_equal(sat_idx, sat_idx_ref[d])

        # two-dimensional control
        sat_idx = utilities.find_saturated(u, lb, ub)
        np.testing.assert_array_equal(sat_idx, sat_idx_ref)

    # Test vector bounds and (n_controls, 1) matrix bounds
    if lb is not None or ub is not None:
        # Construct vector and matrix bounds
        lb_vec = np.full(n_controls, lb) if lb is not None else None
        ub_vec = np.full(n_controls, ub) if ub is not None else None

        lb_mat = np.full((n_controls, 1), lb) if lb is not None else None
        ub_mat = np.full((n_controls, 1), ub) if ub is not None else None

        for u in (u_unsat, u_sat):
            # Test at single points
            for k in range(n_points):
                sat_idx = utilities.find_saturated(u[:, k], lb_vec, ub_vec)
                np.testing.assert_array_equal(sat_idx, sat_idx_ref[:, k])

                sat_idx = utilities.find_saturated(u[:, k], lb_mat, ub_mat)
                np.testing.assert_array_equal(sat_idx, sat_idx_ref[:, k])

                sat_idx = utilities.find_saturated(u[:, k:k+1], lb_vec, ub_vec)
                np.testing.assert_array_equal(sat_idx, sat_idx_ref[:, k:k+1])

                sat_idx = utilities.find_saturated(u[:, k:k+1], lb_mat, ub_mat)
                np.testing.assert_array_equal(sat_idx, sat_idx_ref[:, k:k+1])

            # Test at all points
            sat_idx = utilities.find_saturated(u, lb_vec, ub_vec)
            np.testing.assert_array_equal(sat_idx, sat_idx_ref)

            sat_idx = utilities.find_saturated(u, lb_mat, ub_mat)
            np.testing.assert_array_equal(sat_idx, sat_idx_ref)


@pytest.mark.parametrize('n_points', range(4))
@pytest.mark.parametrize('n_states', range(1, 4))
@pytest.mark.parametrize('n_out', range(4))
def test_approx_derivative(n_states, n_out, n_points):
    if n_points == 0:
        x = rng.uniform(low=-1., high=1., size=(n_states,))
    else:
        x = rng.uniform(low=-1., high=1., size=(n_states, n_points))

    w = np.pi * np.arange(1, n_states+1)
    if n_points > 0:
        w = w.reshape(-1, 1)

    if n_out == 0:
        n_out = 1
        flatten = True
    else:
        flatten = False

    def vector_fun(x):
        f = [np.cos(i * (x * w).sum(axis=0)) for i in range(1, n_out+1)]
        f = np.stack(f, axis=0)
        if flatten:
            return f[0]
        else:
            return f

    f0 = vector_fun(x)

    if n_points == 0:
        if flatten:
            assert f0.ndim == 0
        else:
            assert f0.shape == (n_out,)
    else:
        if flatten:
            assert f0.shape == (n_points,)
        else:
            assert f0.shape == (n_out, n_points)

    # Construct analytical derivatives for comparison
    if n_points == 0:
        dfdx_expected = np.empty((n_out, n_states))
    else:
        dfdx_expected = np.empty((n_out, n_states, n_points))
    for i in range(1, n_out+1):
        dfdx_expected[i-1] = -i * np.sin(i * (x * w).sum(axis=0)) * w
    if flatten:
        dfdx_expected = dfdx_expected[0]

    for method in ['2-point', '3-point', 'cs']:
        dfdx_approx = utilities.approx_derivative(vector_fun, x, method=method)
        np.testing.assert_allclose(dfdx_approx, dfdx_expected,
                                   rtol=1e-03, atol=1e-06)
