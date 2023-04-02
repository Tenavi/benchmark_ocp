"""
The `problem` module is the heart of the `optimalcontrol` package. It implements
the `OptimalControlProblem` class which serves as a standard template for
subclasses implementing specific optimal control problems (OCPs). The
`LinearQuadraticProblem` class is a simple example of how one might subclass
`OptimalControlProblem` to realize a prototypical OCP.

---

* [`OptimalControlProblem`](problem/problem#OptimalControlProblem):
    Base superclass used to implement OCPs.

* [`LinearQuadraticProblem`](problem/linear_quadratic#LinearQuadraticProblem):
    `OptimalControlProblem` implementing the prototypical linear quadratic
    regulator (LQR) OCP.

* [`ProblemParameters`](problem/parameters#ProblemParameters):
    Class housing dynamics and cost function parameters for
    `OptimalControlProblem` instances. Also enables parameter updates.
"""

from .problem import OptimalControlProblem, ProblemParameters
from .linear_quadratic import LinearQuadraticProblem
