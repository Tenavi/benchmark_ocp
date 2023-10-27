import pytest
import numpy as np

from optimalcontrol.open_loop.solutions import CombinedSolution
from optimalcontrol.open_loop.direct.solutions import DirectSolution


rng = np.random.default_rng()


def _make_dummy_sol(n_t=51):
    t = np.linspace(0., 10., n_t)
    x = np.stack((np.cos(t), np.sin(t), np.ones(n_t))) / (t.reshape(1, -1) + 1)
    u = np.stack((-np.sin(t), np.cos(t)))
    p = - x
    v = np.exp(-t)

    return t, x, u, p, v


def test_combining_single_sol():
    t, x, u, p, v = _make_dummy_sol()

    status = 123
    message = "hello world"

    sol = DirectSolution(t, x, u, p, v, status, message)
    wrapped_sol = CombinedSolution(sol)

    assert wrapped_sol._sols[0] is sol
    assert wrapped_sol._t_break.size == 0
    assert np.isinf(wrapped_sol._t_break_extended)

    for key in ['t', 'x', 'u', 'p', 'v', 'status', 'message']:
        np.testing.assert_array_equal(getattr(sol, key),
                                      getattr(wrapped_sol, key))

    t_test = rng.uniform(t[0] - 1, t[-1] + 1, 1000)

    for arr1, arr2 in zip(wrapped_sol(t_test), sol(t_test)):
        np.testing.assert_array_equal(arr1, arr2)


@pytest.mark.parametrize('n_segments', (2, 3, 7))
def test_combining_multiple_sols(n_segments):
    sols = []
    for k in range(n_segments):
        t, x, u, p, v = _make_dummy_sol()

        status = k + 1
        message = f"hello world ({status})"

        sols.append(DirectSolution(t, x, u, p, v, status, message))

    t_break = []
    for k in range(n_segments - 1):
        if k == 0:
            t0 = 0.
        else:
            t0 = t_break[k - 1]
        t_break.append(rng.uniform(t0, t0 + sols[k].t[-1]))

    wrapped_sol = CombinedSolution(sols, t_break=t_break)

    for k in range(n_segments):
        assert wrapped_sol._sols[k] is sols[k]
    np.testing.assert_array_equal(wrapped_sol._t_break.flatten(), t_break)
    np.testing.assert_array_equal(wrapped_sol._t_break_extended[:-1].flatten(),
                                  t_break)
    assert np.isinf(wrapped_sol._t_break_extended[-1])

    assert wrapped_sol.status == n_segments
    assert wrapped_sol.message == f"hello world ({n_segments})"

    # First check that at least time and value function vectors are sorted
    np.testing.assert_array_equal(wrapped_sol.t, np.sort(wrapped_sol.t))
    np.testing.assert_array_equal(wrapped_sol.v, np.sort(wrapped_sol.v)[::-1])
    np.testing.assert_array_less(0., wrapped_sol._v_diff)

    for k in range(n_segments - 2):
        assert wrapped_sol._v_diff[k] > wrapped_sol._v_diff[k + 1]

    # Check that time points are constructed correctly, and each solution is
    # associated with the correct part of the overall time vector
    t0 = 0.
    for k, t1 in enumerate(wrapped_sol._t_break_extended):
        # Manually figure out the bin indices
        idx = np.logical_and(t0 <= wrapped_sol.t, wrapped_sol.t < t1)

        t_segment = wrapped_sol.t[idx] - t0
        n_t = t_segment.size

        np.testing.assert_allclose(t_segment, sols[k].t[:n_t],
                                   rtol=1e-14, atol=1e-14)
        for key in ['x', 'u', 'p']:
            np.testing.assert_allclose(getattr(sols[k], key)[:, :n_t],
                                       getattr(wrapped_sol, key)[:, idx],
                                       rtol=1e-14, atol=1e-14)

        v_expect = sols[k].v[:n_t]
        if k < n_segments - 1:
            v_expect = v_expect + np.maximum(wrapped_sol._v_diff[k], 0.)
        np.testing.assert_allclose(wrapped_sol.v[idx], v_expect,
                                   rtol=1e-14, atol=1e-14)
        t0 = t1

    t0 = 0.
    for k, t1 in enumerate(wrapped_sol._t_break_extended.flatten()):
        if np.isinf(t1):
            t1 = t0 * 2.
        t_test = np.linspace(t0, t0 + 0.99 * (t1 - t0), 10)

        x, u, p, v = wrapped_sol(t_test)
        x_expect, u_expect, p_expect, v_expect = sols[k](t_test - t0)
        if k < n_segments - 1:
            v_expect += np.maximum(wrapped_sol._v_diff[k], 0.)

        np.testing.assert_allclose(x, x_expect, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(u, u_expect, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(p, p_expect, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(v, v_expect, rtol=1e-14, atol=1e-14)

        t0 = t1
