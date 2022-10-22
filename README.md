# benchmark_ocp

This software repository contains a collection of benchmark optimal feedback control problems. Some of these problems are described in

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863)
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415)

If you use this software, please cite one or more of the above works. Please reach out with any questions, or if you encounter bugs or other problems.

## Installation

To install the `ocp` package (in developer mode) run `pip install -e .` from the command line. This package has been developed and tested with the following software dependencies:

    scipy>=1.5.2
    numpy>=1.19.1
    pandas>=1.4.4
    tqdm>=4.64.1
    pytest>=6.1.1
    matplotlib>=3.1.2
    pylgr

The `pylgr` package can be downloaded at [https://github.com/Tenavi/PyLGR](https://github.com/Tenavi/PyLGR).
