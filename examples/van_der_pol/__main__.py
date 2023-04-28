import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

import optimalcontrol as oc

from examples.van_der_pol import VanDerPol
from examples.van_der_pol import example_config as config
from examples import example_utils


# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', int(time.time()))
rng = np.random.default_rng(random_seed + 1)
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
x0_pool = ocp.sample_initial_conditions(config.n_train + config.n_test,
                                        distance=config.x0_distance)

# Warm start the optimal control solver by integrating the system with LQR
args = (oc.simulate.integrate_to_converge, config.t_int, config.t_max)
data, success = example_utils.monte_carlo(ocp, lqr, x0_pool, *args,
                                          **config.sim_kwargs)

for i, sim in enumerate(data):
    # If the simulation failed to converge to equilibrium, reset the guess to
    # interpolate initial and final conditions
    if not success[i]:
        x0 = x0_pool[:, i:i+1]
        x_interp = interp1d([0., config.t_int], np.hstack((x0, ocp.xf)))
        sim['t'] = np.linspace(0., config.t_int, 100)
        sim['x'] = x_interp(sim['t'])
        sim['u'] = lqr(sim['x'])
    # Use LQR to generate a guess for the costates
    sim['p'] = 2. * lqr.P @ (sim['x'] - ocp.xf)
    # Use the closed-loop cost-to-go as a guess for the value function
    sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]

# Solve open loop optimal control problems
data, status, messages = example_utils.generate_from_guess(
    ocp, data, **config.open_loop_kwargs)

# Reserve a subset of data for testing and use the rest for training
data = np.asarray(data, dtype=object)[status == 0]
rng.shuffle(data)
test_data = data[:config.n_test]
train_data = data[config.n_test:]

print("Training SVR-based supervised learning controller...")

# Turn data into numpy arrays for training and test evaluation
_, x_train, u_train, _, _ = oc.utilities.stack_dataframes(*train_data)
_, x_test, u_test, _, _ = oc.utilities.stack_dataframes(*test_data)

svr_controller = example_utils.SVRController(x_train, u_train,
                                             u_lb=ocp.parameters.u_lb,
                                             u_ub=ocp.parameters.u_ub,
                                             **config.controller_kwargs)

u_train_pred = svr_controller(x_train)
u_test_pred = svr_controller(x_test)

plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(x_train[0], x_train[1], u_train.flatten(), marker='o', alpha=0.25)
ax.scatter(x_train[0], x_train[1], u_train_pred.flatten(), marker='x', alpha=0.25)

plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(x_test[0], x_test[1], u_test.flatten(), marker='o', alpha=0.25)
ax.scatter(x_test[0], x_test[1], u_test_pred.flatten(), marker='x', alpha=0.25)

plt.show()

