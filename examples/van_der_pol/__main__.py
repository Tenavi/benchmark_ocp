import numpy as np
import pandas as pd
import os
import time
from sklearn.svm import SVR
from matplotlib import pyplot as plt

import optimalcontrol as oc

from examples.van_der_pol import VanDerPol
from examples.van_der_pol import example_config as config
from examples.example_utils import monte_carlo, generate_from_guess


# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', int(time.time()))
ocp = VanDerPol(x0_sample_seed=random_seed, **config.params)
xf = ocp.xf.flatten()
uf = ocp.uf.flatten()

# Create an LQR controller as a baseline
# System matrices (vector field Jacobians)
A, B = ocp.jac(xf, uf)
# Cost matrices (1/2 Running cost Hessians)
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

# Warm start the optimal control solver by integrating the system with LQR
args = (oc.simulate.integrate_to_converge, config.t_int, config.t_max)
train_data, train_success = monte_carlo(ocp, lqr, x0_train, *args,
                                        **config.sim_kwargs)
test_data, test_success = monte_carlo(ocp, lqr, x0_test, *args,
                                      **config.sim_kwargs)

for sim in train_data + test_data:
    # Use LQR to generate a guess for the costates
    sim['p'] = 2. * lqr.P @ (sim['x'] - ocp.xf)
    # Use the closed-loop cost-to-go as a guess for the value function
    sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]

train_data, unsolved, success = generate_from_guess(ocp, train_data,
                                                    **config.open_loop_kwargs)

