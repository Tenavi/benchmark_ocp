import pytest

import numpy as np

from optimalcontrol import utilities


rng = np.random.default_rng()

@pytest.mark.parametrize("n", [0., 1.5, np.array([[5],[6]]), [7,8], "n"])
def test_check_int_input_bad_type(n):
    """Make sure `check_int_input` catches a range of bad input types."""
    with pytest.raises(TypeError):
        utilities.check_int_input(n, "n")

@pytest.mark.parametrize("min", [-7., np.pi, np.array([[1,2]]), [3,4], "m"])
def test_check_int_input_bad_min(min):
    """Make sure `check_int_input` catches a range of bad `min` input types."""
    with pytest.raises(TypeError):
        utilities.check_int_input(rng.choice(100), "n", min=min)

@pytest.mark.parametrize("shape", [(),(1,),(1,1)])
def test_check_int_input(shape):
    """Make sure `check_int_input` works with a variety of acceptable inputs."""
    n = rng.choice(100, size=shape) - 50
    assert utilities.check_int_input(n.tolist(), "n") == int(n)
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        assert utilities.check_int_input(n.astype(dtype), "n") == int(n)

@pytest.mark.parametrize("min", [0, 1, -5, 10])
def test_check_int_input_with_min(min):
    """Make sure `check_int_input` works with minimum inputs and throws an error
    if the desired minimum is not found."""
    assert utilities.check_int_input(min, "n", min=min) == min
    assert utilities.check_int_input(min + 1, "n", min=min) == min + 1
    with pytest.raises(ValueError):
        utilities.check_int_input(min - 1, "n", min=min)

@pytest.mark.parametrize("n_rows", [1, 2, 3, -1])
def test_resize_vector(n_rows):
    if n_rows > 0:
        x = rng.normal(size=n_rows)
    else:
        x = rng.normal(size=17)
    y_expect = x.reshape(-1,1)

    y = utilities.resize_vector(x, n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.reshape(-1,1), n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.reshape(1,-1), n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x.tolist(), n_rows)
    np.testing.assert_allclose(y, y_expect)

    if n_rows > 2:
        with pytest.raises(ValueError):
            y = utilities.resize_vector(x[:-1], n_rows)

@pytest.mark.parametrize("n_rows", [2, 3])
def test_resize_vector_float_in(n_rows):
    x = np.full(n_rows, rng.normal())
    y_expect = x.reshape(-1,1)

    y = utilities.resize_vector(x[0], n_rows)
    np.testing.assert_allclose(y, y_expect)

    y = utilities.resize_vector(x[:1], n_rows)
    np.testing.assert_allclose(y, y_expect)

@pytest.mark.parametrize("n_points", range(4))
@pytest.mark.parametrize("n_states", range(1,4))
@pytest.mark.parametrize("n_out", range(4))
def test_approx_derivative(n_states, n_out, n_points):
    if n_points == 0:
        x = rng.uniform(low=-1., high=1., size=(n_states,))
    else:
        x = rng.uniform(low=-1., high=1., size=(n_states, n_points))

    w = np.pi * np.arange(1,n_states+1)
    if n_points > 0:
        w = w.reshape(-1, 1)

    if n_out == 0:
        n_out = 1
        flatten = True
    else:
        flatten = False

    def vector_fun(x):
        f = [np.cos(i * (x * w).sum(axis=0)) for i in range(1,n_out+1)]
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
    for i in range(1,n_out+1):
        dfdx_expected[i-1] = -i * np.sin(i * (x * w).sum(axis=0)) * w
    if flatten:
        dfdx_expected = dfdx_expected[0]

    for method in ["2-point", "3-point", "cs"]:
        dfdx_approx = utilities.approx_derivative(vector_fun, x, method=method)
        np.testing.assert_allclose(
            dfdx_approx, dfdx_expected, rtol=1e-03, atol=1e-06
        )
