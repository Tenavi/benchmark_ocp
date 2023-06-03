"""
The `problem` module is the heart of the `optimalcontrol` package. It implements
the `OptimalControlProblem` class which serves as a standard template for
subclasses implementing specific optimal control problems (OCPs). The
`LinearQuadraticProblem` class is a simple example of how one might subclass
`OptimalControlProblem` to realize a prototypical OCP. The architecture is
designed to discourage hard-coding of any dynamics or cost function parameters
directly inside the `OptimalControlProblem` class; instead these are stored in
a `ProblemParameters` instance which is attached to the class instance. The
`ProblemParameters` is initialized with default variables attached to the class,
and can be updated once initialized (e.g. for testing robustness during closed-
loop simulation).

---

* [`OptimalControlProblem`](problem/problem#OptimalControlProblem):
    Base superclass used to implement OCPs.

* [`LinearQuadraticProblem`](problem/linear_quadratic#LinearQuadraticProblem):
    `OptimalControlProblem` implementing the prototypical linear quadratic
    regulator (LQR) OCP.

* [`ProblemParameters`](problem/problem#ProblemParameters):
    Class housing dynamics and cost function parameters for
    `OptimalControlProblem` instances.
"""

from .problem import OptimalControlProblem, ProblemParameters
from .linear_quadratic import LinearQuadraticProblem
