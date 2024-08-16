"""
The `analyze` module contains tools for linearization and linear stability and
robustness analysis. It is still under development.

---

* [`find_equilibrium`](analyze/linear#find_equilibrium):
    Search for stable and unstable closed-loop equilibrium points using
    integration.

* [`linear_stability`](analyze/linear#linear_stability):
    Compute eigenvalues for the closed-loop system linearized at an equilibrium.

* [`disk_margins`](analyze/robustness#disk_margins):
    Compute and plot disk margins for the closed-loop system linearized at an
    equilibrium. Disk margins generalize classic gain and phase margins to
    simultaneous gain and phase perturbations in multiple channels of a MIMO
    control system.
"""

from .linear import find_equilibrium, linear_stability
from .robustness import disk_margins
