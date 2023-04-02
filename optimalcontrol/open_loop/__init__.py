"""
The `open_loop` module contains functions to solve open-loop optimal control
problems (OCPs) for individual initial conditions. There are of course a wide
variety of algorithms to solve such problems, and this module cannot hope to
implement them all. Rather, it provides simple interfaces to some standard
approaches for the problem classes represented in `optimalcontrol`. This module
is intended to be a starting point for generating data for training and testing
optimal feedback control laws.

The methods implemented in `open_loop` are classified as "direct" or "indirect".
These are available through the [`open_loop.direct`](open_loop/direct) and
[`open_loop.indirect`](open_loop/indirect) submodules, respectively, or through
the `method` keyword argument in functions callable directly from `open_loop`.

The direct methods implemented in this module use pseudospectral collocation to
transform the open-loop OCP into a constrained optimization problem, which is
then solved using sequential least squares quadratic programming (SLSQP). See
ref. [2] for more details.

The indirect method is to solve the two-point boundary value problem (BVP)
arising from Pontryagin's Maximum Principle (PMP), which provides necessary
conditions for OCPs. This BVP is solved using `scipy.integrate.solve_bvp`. See
ref. [1] for more details.

The direct method is generally considered to be more robust than the indirect
method, which is known to be highly sensitive to the initial guess for the
costates. On the other hand, when successful, the indirect method often yields
more accurate results and can also be faster.

In general, the performance of both approaches depends on the problem scaling,
problem complexity, and the quality of the initial guess provided to the solver.
More information and practical considerations are given in refs. [1-3] below.

##### References

1. W. Kang, Q. Gong, T. Nakamura-Zimmerer, and F. Fahroo, *Algorithms of
  data development for deep learning and feedback design: A survey,* Physica D:
  Nonlinear Phenomena  (2021), pp. 132955.
  https://doi.org/10.1016/j.physd.2021.132955
2. I. M. Ross, *A Primer on Pontryagin’s Principle in Optimal Control,*
  Collegiate Publishers, San Francisco, CA, 2nd ed., 2015.
3. I. M. Ross and M. Karpenko, *A review of pseudospectral optimal control: From
  theory to flight,* Annual Reviews in Control, 36 (2012), pp. 182–197,
  https://doi.org/10.1016/j.arcontrol.2012.09.002

---

##### Unified interface

* [`solve_fixed_time`](open_loop/solve#solve_fixed_time):
    Solve a single open-loop OCP with a fixed, finite time horizon.

* [`solve_infinite_horizon`](open_loop/solve#solve_infinite_horizon):
    Solve a single open-loop infinite horizon OCP.

* [`solutions.OpenLoopSolution`](open_loop/solutions#OpenLoopSolution)
    Methods implemented in this model return open-loop OCP solutions as
    instances of this class.

---

##### Direct methods: [`direct`](open_loop/direct)

* [`direct.solve_fixed_time`](open_loop/direct#solve_fixed_time):
    Use a pseudospectral method to solve a single open-loop OCP with a fixed,
    finite time horizon.

* [`direct.solve_infinite_horizon`](open_loop/direct#solve_infinite_horizon):
    Use a pseudospectral method to solve a single open-loop infinite horizon
    OCP.

---

##### Indirect methods: [`indirect`](open_loop/indirect)

* [`indirect.solve_fixed_time`](open_loop/indirect#solve_fixed_time):
    Solve a single open-loop OCP with a fixed, finite time horizon via PMP.

* [`direct.solve_infinite_horizon`](open_loop/direct#solve_infinite_horizon):
    Solve a series of finite horizon OCPs via PMP to approximate the solution to
    a single infinite horizon OCP.
"""

from .solve import *
