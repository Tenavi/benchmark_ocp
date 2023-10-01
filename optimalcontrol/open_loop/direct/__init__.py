"""
This submodule implements a direct method for open-loop optimal control, based
on Legendre Gauss Radau (LGR) pseudospectral collocation. The implementation
follows the methodology proposed in refs. [1, 2].

##### Functions

* [`solve_fixed_time`](direct/solve#solve_fixed_time):
    Use a pseudospectral method to solve a single open-loop OCP with a fixed,
    finite time horizon.

* [`solve_infinite_horizon`](direct/solve#solve_infinite_horizon):
    Use a pseudospectral method to solve a single open-loop infinite horizon
    OCP.

##### Submodules

* [`Radau`](direct/radau):
    Functions to construct the LGR collocation points, differentiation matrix,
    and integration weights, and map from physical time in [0, inf) to the
    half-open interval [-1, 1).

* [`utilities`](direct/utilities):
    Functions used to set up a finite-dimensional constrained optimization
    problem from an infinite-dimensional `OptimalControlProblem` and a
    collocation scheme generated using the `radau` submodule.

##### References

1. I. M. Ross, Q. Gong, F. Fahroo, and W. Kang, *Practical stabilization through
    real-time optimal control*, in American Control Conference, 2006, pp.
    304-309. https://doi.org/10.1109/ACC.2006.1655372
2. F. Fahroo and I. M. Ross, *Pseudospectral methods for infinite-horizon
    nonlinear optimal control problems*, Journal of Guidance, Control, and
    Dynamics, 31 (2008), pp. 927-936. https://doi.org/10.2514/1.33117
"""

from .solve import *
