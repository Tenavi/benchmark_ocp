import pytest

import numpy as np
from matplotlib import pyplot as plt

from optimalcontrol.simulate import integrate_closed_loop, integrate_to_converge
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params

def test_integrate_closed_loop_LQR():
    """Basic test of an LQR-controlled linear system, which should be stable."""
    n_states = 3
    n_controls = 2
    tspan = [0.,20.]

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(
        A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, x0_lb=-1., x0_ub=1.
    )
    LQR = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf)

    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    t, x, _ = integrate_closed_loop(ocp, LQR, tspan, x0, atol=1e-12, rtol=1e-06)
    u = LQR(x)
    cost = ocp.running_cost(x, u)

    # At large final time, system should be close to equilibrium
    np.testing.assert_allclose(t[-1], tspan[-1])
    np.testing.assert_allclose(x[:,-1], xf.flatten(), atol=1e-04, rtol=1e-02)
    np.testing.assert_allclose(u[:,-1], uf.flatten(), atol=1e-04, rtol=1e-02)
    np.testing.assert_allclose(cost[-1], 0., atol=1e-08)

    # Expect integrated cost to be close to LQR value function
    xPx = (x[:,:1] - xf).T @ LQR.P @ (x[:,:1] - xf)
    J = ocp.total_cost(t, x, u)[-1]
    np.testing.assert_allclose(xPx, J, atol=1e-02, rtol=1e-02)