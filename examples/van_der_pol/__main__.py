import os
import time
import argparse as ap

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from optimalcontrol import controls, simulate, open_loop, utilities, analysis

from examples.van_der_pol import VanDerPol
from examples.van_der_pol import example_config as config
from examples import example_utilities as example_utils


parser = ap.ArgumentParser()
parser.add_argument('-s', '--show_plots', action='store_true',
                    help="Show plots at runtime, in addition to saving")
args = parser.parse_args()

# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', None)
if random_seed is None:
    random_seed = int(time.time())
rng = np.random.default_rng(random_seed + 1)

ocp = VanDerPol(x0_sample_seed=random_seed, **config.params)
xf = ocp.xf.flatten()
uf = ocp.uf.flatten()

# Create an LQR controller as a baseline
# System matrices (vector field Jacobians)
A, B = ocp.jac(xf, uf)
# Cost matrices (1/2 Running cost Hessians)
Q, R = ocp.running_cost_hess(xf, uf)
lqr = controls.LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R,
                                        u_lb=ocp.parameters.u_lb,
                                        u_ub=ocp.parameters.u_ub, xf=xf, uf=uf)

# Generate some training and test data

# First sample initial conditions
x0_pool = ocp.sample_initial_conditions(config.n_train + config.n_test,
                                        distance=config.x0_distance)

# Warm start the optimal control solver by integrating the system with LQR
lqr_sims, status = simulate.monte_carlo_to_converge(
    ocp, lqr, x0_pool, config.t_int, config.t_max, **config.sim_kwargs)
lqr_sims = np.asarray(lqr_sims, dtype=object)

for i, sim in enumerate(lqr_sims):
    # If the simulation failed to converge to equilibrium, reset the guess to
    # interpolate initial and final conditions
    if status[i] != 0:
        x0 = x0_pool[:, i:i+1]
        x_interp = interp1d([0., config.t_int], np.hstack((x0, ocp.xf)))
        sim['t'] = np.linspace(0., config.t_int, 100)
        sim['x'] = x_interp(sim['t'])
        sim['u'] = lqr(sim['x'])
    # Use LQR to generate a guess for the costates
    sim['p'] = 2. * lqr.P @ (sim['x'] - ocp.xf)

# Solve open loop optimal control problems
data, status, messages = example_utils.generate_data(ocp, lqr_sims,
                                                     **config.open_loop_kwargs)

# Reserve a subset of data for testing and use the rest for training
data_idx = np.arange(x0_pool.shape[1])[status == 0]
rng.shuffle(data_idx)
train_idx = data_idx[config.n_test:]
test_idx = data_idx[:config.n_test]

train_data = data[train_idx]
test_data = data[test_idx]

# Turn data into numpy arrays for training and test evaluation
_, x_train, u_train, _, _ = utilities.stack_dataframes(*train_data)
_, x_test, u_test, _, _ = utilities.stack_dataframes(*test_data)

controller = example_utils.NNController(x_train, u_train,
                                        u_lb=ocp.parameters.u_lb,
                                        u_ub=ocp.parameters.u_ub,
                                        random_state=random_seed + 2,
                                        **config.controller_kwargs)

train_r2 = r2_score(u_train.T, controller(x_train).T)
test_r2 = r2_score(u_test.T, controller(x_test).T)

print(f"\nR2 score: {train_r2:.4f} (train) {test_r2:.4f} (test)")

# Evaluate performance of the NN controller in closed-loop simulation
sims, status = simulate.monte_carlo_to_converge(
    ocp, controller, x0_pool, config.t_int, config.t_max, **config.sim_kwargs)

for sim in sims:
    sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]

# If the NN cost is lower than the optimal cost, we may have found a better
# local minimum
for dataset, idx in zip((train_data, test_data), (train_idx, test_idx)):
    for i, sol in enumerate(dataset):
        sim = sims[idx[i]]
        if sol['v'][0] > sim['v'][0]:
            # Try to resolve the OCP if the initial guess looks better
            new_sol = open_loop.solve_infinite_horizon(
                ocp, sim['t'], sim['x'], u=sim['u'],
                p=2. * lqr.P @ (sim['x'] - ocp.xf), v=sim['v'],
                **config.open_loop_kwargs)
            if new_sol.v[0] < sol['v'][0]:
                print(f"Found a better solution for OCP #{idx[i]:d} using NN "
                      f"warm start")
                for key in sol.keys():
                    sol[key] = getattr(new_sol, key)

# Evaluate the linear stability of the learned controller
x, f = analysis.find_equilibrium(ocp, controller, xf)
jac = utilities.closed_loop_jacobian(x, ocp.jac, controller)
eigs, max_eig = analysis.linear_stability(jac)

# If an unstable equilibrium was found, try perturbing the equilibrium slightly
# and integrating from there to find a stable
if max_eig > 0.:
    x += rng.normal(scale=1/100, size=x.shape)
    t, x, status = simulate.integrate_to_converge(
        ocp, controller, x, config.t_int, config.t_max, **config.sim_kwargs)

    print("Closed-loop integration from the unstable equilibrium led to:")
    print(x[:, -1:])
    print("Retrying root-finding...")

    x, f = analysis.find_equilibrium(ocp, controller, x[:, -1])
    jac = utilities.closed_loop_jacobian(x, ocp.jac, controller)
    eigs, max_eig = analysis.linear_stability(jac)

# Plot the results
figs = {'training': dict(), 'test': dict()}
for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
    lqr_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                 for sim in lqr_sims[data_idx]]
    nn_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                for sim in sims[data_idx]]
    figs[data_name]['cost_comparison'] = example_utils.plot_total_cost(
        [sol['v'][0] for sol in data[data_idx]],
        controller_costs={'LQR': lqr_costs, 'NN control': nn_costs},
        title=f'Closed-loop cost evaluation ({data_name})')
    figs[data_name]['closed_loop_3d'] = example_utils.plot_closed_loop_3d(
        sims[data_idx], data[data_idx], controller_name='NN control',
        title=f'Closed-loop trajectories and controls ({data_name})')
    figs[data_name]['closed_loop'] = example_utils.plot_closed_loop(
        ocp, sims[data_idx], data_name=data_name)

# Save data and figures
data_dir = os.path.join('examples', 'van_der_pol', 'data')
fig_dir = os.path.join('examples', 'van_der_pol', 'figures')

os.makedirs(data_dir, exist_ok=True)

example_utils.save_data(train_data, os.path.join(data_dir, 'train.csv'))
example_utils.save_data(test_data, os.path.join(data_dir, 'test.csv'))

for data_name, figs_subset in figs.items():
    _fig_dir = os.path.join(fig_dir, data_name)
    os.makedirs(_fig_dir, exist_ok=True)
    for fig_name, fig in figs_subset.items():
        plt.figure(fig)
        plt.savefig(os.path.join(_fig_dir, fig_name + '.pdf'))

if args.show_plots:
    plt.show()
