# benchmark_ocp

##### Authors
* Tenavi Nakamura-Zimmerer (tenavi.nz@gmail.com)
* Jiequn Han (jhan@flatironinstitute.org)
* Qi Gong (qgong@ucsc.edu)
* Wei Kang (wkang@nps.edu)

*This software is being developed independently of NASA. It is not endorsed or supported by NASA or the US government.*

The `optimalcontrol` package is a framework for describing optimal control problems (OCPs) in python.

A collection of some benchmark OCPs of varying difficulty are located in the `examples` folder, separate from the `optimalcontrol` package. Some of these OCPs are described in

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863), T. Nakamura-Zimmerer, Q. Gong, and W. Kang, 2022.
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415), T. Nakamura-Zimmerer, Q. Gong, and W. Kang, 2021.
  * [Pseudospectral methods for infinite-horizon nonlinear optimal control problems](https://doi.org/10.2514/1.33117), F. Fahroo and I. M. Ross, 2008.

If you use this software, please cite the software package and the relevant publication(s). Please reach out with any questions, or if you encounter bugs or other problems.

---

# Installation

First create a python environment (using e.g. conda or pip) with

    python>=3.8

Then to install the `optimalcontrol` package (in developer mode), run

    pip install -e .

This package and the examples have been developed and tested with the following software
dependencies:
    
    numpy>=1.17
    scipy>=1.8
    pytest
    jupyter
    matplotlib
    pandas
    scikit-learn>=1.0
    tqdm

---

## Test

From the root directory, run

    pytest tests -s -v

---

## Generate documentation

Install `pdoc` and run

    pdoc optimalcontrol --d numpy --math -t docs/.template/ -o docs/optimalcontrol
    pdoc examples --d numpy --math -t docs/.template/ -o docs/examples

---

# The `optimalcontrol` package

The `optimalcontrol` package is made up of the following modules:

* `problem`: The most import piece of the package. Contains the `OptimalControlProblem` base superclass used to implement OCPs.

* `controls`: Contains the `Controller` template class for implementing feedback control policies.

* `open_loop`: Basic functions to solve open-loop OCPs for individual initial conditions, for the purpose of data generation. Difficult problems may require custom algorithms.

* `simulate`: Functions to integrate closed-loop dynamical systems, facilitating performance and stability testing of feedback control laws.

* `sampling`: Contains frameworks for implementing algorithms to sample the state space for data generation and controller testing.

* `analyze`: Tools for linearization and linear closed-loop stability analysis. In development.

* `utilities`: General utility functions.

---

# The benchmark `examples`
