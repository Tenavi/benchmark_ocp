This software is being developed independently of NASA. It is not endorsed or supported by NASA or the US government.

---

This software repository contains a collection of benchmark optimal feedback control problems. Some of these problems are described in

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863)
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415)

If you use this software, please cite the software package and/or one or more of the above works. Please reach out with any questions, or if you encounter bugs or other problems.

## Installation

To install the `optimalcontrol` package (in developer mode) run `pip install -e .`
from the command line. This package has been developed and tested with the
following software dependencies:

    python>=3.7
    numpy>=1.21.6
    scipy>=1.7.3
    pytest
    jupyter
    tqdm>=4.65.0
    pandas>=1.3.5
    scikit-learn>=1.0.2
    matplotlib>=3.5.3
    pylgr

The `pylgr` package can be downloaded at https://github.com/Tenavi/PyLGR.

### Test

Run `pytest tests -s -v` from the root directory.

### Generate documentation

Install `pdoc` and run

    pdoc optimalcontrol --d numpy --math -t documentation/.template/ -o documentation/