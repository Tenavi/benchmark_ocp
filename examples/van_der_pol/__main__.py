import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import optimalcontrol as oc

from examples.van_der_pol import VanDerPol
from examples.van_der_pol import example_config as config
from examples import example_utils


# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', None)
if random_seed is None:
    random_seed = int(time.time())
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
sim_args = (oc.simulate.integrate_to_converge, config.t_int, config.t_max)
lqr_sims, success = example_utils.monte_carlo(ocp, lqr, x0_pool, *sim_args,
                                              **config.sim_kwargs)
lqr_sims = np.asarray(lqr_sims, dtype=object)

for i, sim in enumerate(lqr_sims):
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
    ocp, lqr_sims, **config.open_loop_kwargs)

# Reserve a subset of data for testing and use the rest for training
data_idx = np.arange(x0_pool.shape[1])
np.random.default_rng(random_seed + 1).shuffle(data_idx)
train_idx = data_idx[status == 0][config.n_test:]
test_idx = data_idx[status == 0][:config.n_test]

data = np.asarray(data, dtype=object)
train_data = data[train_idx]
test_data = data[test_idx]

# Turn data into numpy arrays for training and test evaluation
_, x_train, u_train, _, _ = oc.utilities.stack_dataframes(*train_data)
_, x_test, u_test, _, _ = oc.utilities.stack_dataframes(*test_data)

controller = example_utils.NNController(x_train, u_train,
                                        u_lb=ocp.parameters.u_lb,
                                        u_ub=ocp.parameters.u_ub,
                                        random_state=random_seed+2,
                                        **config.controller_kwargs)

train_r2 = r2_score(u_train.T, controller(x_train).T)
test_r2 = r2_score(u_test.T, controller(x_test).T)

print(f"\nR2 score: {train_r2:.4f} (train) {test_r2:.4f} (test)")

for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
    example_utils.evaluate_closed_loop(ocp, controller, x0_pool, data, lqr_sims,
                                       data_idx, data_name, *sim_args,
                                       **config.sim_kwargs)

# Evaluate the linear stability of the learned controller
x, f, jac, eigs, max_eig = oc.utilities.find_equilibrium(ocp, controller, xf, tol=1e-03)

if max_eig > 0.:
    t, x_new, status = oc.simulate.integrate_to_converge(
        ocp, controller, x, *sim_args[1:], **config.sim_kwargs)

    print("Integration of closed-loop system led to:")
    print(x_new[:, -1])
    x, f, jac, eigs, max_eig = oc.utilities.find_equilibrium(
        ocp, controller, x_new[:, -1], tol=1e-03)

plt.show()
