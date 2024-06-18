"""
The `simulate` module contains functions to integrate closed-loop dynamical
systems. It provides a convenient interface between `OptimalControlProblem` and
`Controller` classes with `scipy.integrate.solve_ivp`, facilitating closed-loop
performance and stability testing of feedback control laws.

---

* [`integrate_closed_loop`](simulate/simulate#integrate_closed_loop):
    Integrate a closed-loop system over a fixed time horizon.

* [`integrate_to_converge`](simulate/simulate#integrate_to_converge):
    Integrate a closed-loop system until it converges to an equilibrium.

* [`monte_carlo_closed_loop`](simulate/simulate#monte_carlo_closed_loop):
    Integrate a closed-loop system over a fixed time horizon from multiple
    initial conditions.

* [`monte_carlo_to_converge`](simulate/simulate#monte_carlo_to_converge):
    Integrate a closed-loop system until convergence to an equilbrium from
    multiple initial conditions.
"""

from .simulate import (integrate, integrate_to_converge,
                       monte_carlo, monte_carlo_to_converge)
