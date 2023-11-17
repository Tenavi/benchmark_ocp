# Benchmark problems for optimal control

_This software is being developed independently of NASA. It is not endorsed or supported by NASA or the US government._

This software repository contains a framework for describing optimal control problems (OCPs) in python, and a collection of some benchmark OCPs of varying difficulty. Some of these problems are described in

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863)
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415)

If you use this software, please cite the software package and/or one or more of the above works. Please reach out with any questions, or if you encounter bugs or other problems.

---

## Installation

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

Run `pytest tests -s -v` from the root directory.

---

## Generate documentation

Install `pdoc` and run

    pdoc optimalcontrol --d numpy --math -t documentation/.template/ -o documentation/

---

## Example problems
