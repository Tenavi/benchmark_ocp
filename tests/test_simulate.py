import pytest

import numpy as np

from optimalcontrol.simulate import integrate_closed_loop, integrate_to_converge
from optimalcontrol.problem import LinearQuadraticProblem
from optimalcontrol.controls import LinearQuadraticRegulator

from ._utilities import make_LQ_params

def test_integrate_closed_loop():
    n_states = 3
    n_controls = 2

    A, B, Q, R, xf, uf = make_LQ_params(n_states, n_controls)
    ocp = LinearQuadraticProblem(
        A=A, B=B, Q=Q, R=R, xf=xf, uf=uf, x0_lb=-1., x0_ub=1.
    )
    LQR = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf)