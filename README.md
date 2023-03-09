# benchmark_ocp

This software repository contains a collection of benchmark optimal feedback control problems. Some of these problems are described in

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863)
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415)

If you use this software, please cite one or more of the above works. Please reach out with any questions, or if you encounter bugs or other problems.

## Installation

To install the `optimalcontrol` package (in developer mode) run `pip install -e .`
from the command line. This package has been developed and tested with the
following software dependencies:

    python>=3.7
    numpy>=1.21.6
    scipy>=1.7.3
    pandas>=1.3.5
    tqdm>=4.65.0
    pytest>=7.2.2
    matplotlib>=3.5.3
    pylgr

The `pylgr` package can be downloaded at [https://github.com/Tenavi/PyLGR](https://github.com/Tenavi/PyLGR).

## To-do list

  * Open loop BVP solver
  * Time-dependent problems
  * `run` folder
  * other examples
  * update and publish `pylgr` package
  * Open loop direct solver
  * Open loop solvers for time-dependent problems
  * Constraints and integration events
  * RK4 batch integrator
  * Zero-order hold control wrapper
  * LQR integrator