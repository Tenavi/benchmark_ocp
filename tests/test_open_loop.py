import pytest
import numpy as np

from optimalcontrol.open_loop import indirect
from optimalcontrol.simulate import integrate_closed_loop
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params

@pytest.mark.parametrize("method", ["indirect"])
def test_solve_fixed_time(method):
    """
    Basic test of an LQR-controlled linear system integrated over a fixed time
    horizon. Since the closed-loop system should be stable, checks that the
    system is close to equilibrium after a reasonably long time horizon.
    """
    n_states = 3
    n_controls = 2
    t_span = [0., 30.]

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, x0_lb=-1.,
                                 x0_ub=1., u_lb=-1., u_ub=1.)
    LQR = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, u_lb=-1.,
                                   u_ub=1.)

    x0 = ocp.sample_initial_conditions(n_samples=1, distance=1/2)

    t, x, _ = integrate_closed_loop(ocp, LQR, t_span, x0)
    p = 2. * LQR.P @ (x - xf)



    u = LQR(x)