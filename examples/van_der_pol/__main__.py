import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from sklearn.svm import SVR
from matplotlib import pyplot as plt

import optimalcontrol as oc

from examples.van_der_pol import VanDerPol
from examples.van_der_pol import example_config as config


# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', int(time.time()))
ocp = VanDerPol(x0_sample_seed=random_seed, **config.params)
xf = ocp.xf.flatten()
uf = ocp.uf.flatten()

# Create an LQR controller as a baseline)
A, B = ocp.jac(xf, uf)
Q, R = ocp.running_cost_hess(xf, uf)
lqr = oc.controls.LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R,
                                           u_lb=ocp.parameters.u_lb,
                                           u_ub=ocp.parameters.u_ub,
                                           xf=xf, uf=uf)

# Generate some training and test data

# First sample initial conditions
x0_train = ocp.sample_initial_conditions(config.n_train,
                                         distance=config.x0_distance)
x0_test = ocp.sample_initial_conditions(config.n_test,
                                        distance=config.x0_distance)

# Lists for storing data
train_data = []
test_data = []


