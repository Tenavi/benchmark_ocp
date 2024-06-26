import os
import time

import numpy as np
from scipy.interpolate import make_interp_spline

from optimalcontrol import simulate, utilities, analyze
from optimalcontrol.controls import LinearQuadraticRegulator
from optimalcontrol.open_loop import solve_infinite_horizon

from examples.common_utilities import data_utils, supervised_learning, plotting

from examples.burgers.problem_definition import BurgersPDE, plot_closed_loop
from examples.burgers import example_config as config


# Initialize the optimal control problem
random_seed = getattr(config, 'random_seed', None)
if random_seed is None:
    random_seed = int(time.time())
rng = np.random.default_rng(random_seed + 1)

ocp = BurgersPDE(x0_sample_seed=random_seed, **config.params)
xf = np.zeros(ocp.n_states)
uf = np.zeros(ocp.n_controls)

# Create an LQR controller as a baseline
# System matrices (vector field Jacobians)
A, B = ocp.jac(xf, uf)
# Cost matrices (1/2 Running cost Hessians)
Q, R = ocp.running_cost_hess(xf, uf)
lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R,
                               u_lb=ocp.control_lb, u_ub=ocp.control_ub)

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
        x_interp = make_interp_spline([0., config.t_int],
                                      np.hstack((x0, xf.reshape(-1, 1))),
                                      k=1, axis=1)
        sim['t'] = np.linspace(0., config.t_int, 100)
        sim['x'] = x_interp(sim['t'])
        sim['u'] = lqr(sim['x'])
    # Use LQR to generate a guess for the costates
    sim['p'] = 2. * lqr.P @ sim['x']

# Solve open loop optimal control problems
data, status, messages = data_utils.generate_data(ocp, lqr_sims,
                                                  **config.open_loop_kwargs)

print("\n" + "+" * 80)

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

print("\nTraining neural network controller...")
nn_control = supervised_learning.NeuralNetworkController(
    x_train, u_train, u_lb=ocp.control_lb, u_ub=ocp.control_ub,
    random_state=random_seed + 3, **config.nn_kwargs)

print("\n" + "+" * 80)

for controller in (lqr, nn_control):
    print(f"\nLinear stability analysis for {type(controller).__name__:s}:")

    x = analyze.find_equilibrium(ocp, controller, xf, config.t_int,
                                 config.t_max, **config.sim_kwargs)
    print("Equilibrium point:")
    print(x.reshape(-1, 1))
    analyze.linear_stability(ocp, controller, x)

print("\n" + "+" * 80)

for controller in (lqr, nn_control):
    train_r2 = controller.r2_score(x_train, u_train)
    test_r2 = controller.r2_score(x_test, u_test)
    print(f"\n{type(controller).__name__:s} R2 score: {train_r2:.4f} (train) "
          f"{test_r2:.4f} (test)")

print("\n" + "+" * 80 + "\n")

# Evaluate performance of the learned controllers in closed-loop simulation
nn_sims, _ = simulate.monte_carlo_to_converge(
    ocp, nn_control, x0_pool, config.t_int, config.t_max, **config.sim_kwargs)

for sims in (lqr_sims, nn_sims):
    for sim in sims:
        sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]
        sim['L'] = ocp.running_cost(sim['x'], sim['u'])

# If the closed-loop cost is lower than the optimal cost, we may have found
# a better local minimum
for dataset, idx in zip((train_data, test_data), (train_idx, test_idx)):
    for i, sol in enumerate(dataset):
        sim = nn_sims[idx[i]]

        if sol['v'][0] > sim['v'][0]:
            # Try to resolve the OCP if the initial guess looks better
            new_sol = solve_infinite_horizon(
                ocp, sim['t'], sim['x'], u=sim['u'], v=sim['v'],
                p=2. * lqr.P @ sim['x'],
                **config.open_loop_kwargs)
            cost_change = 1. - new_sol.v[0] / sol['v'][0]
            if cost_change < 0.:
                print(f"Found a better solution for OCP #{idx[i]:d} using "
                      f"warm start with {type(nn_control).__name__:s}.")
                print(f"    Cost improvement = {-100 * cost_change:.2f}%")
                new_sol.L = ocp.running_cost(new_sol.x, new_sol.u)
                for key in sol.keys():
                    sol[key] = getattr(new_sol, key)

        sol['L'] = ocp.running_cost(sol['x'], sol['u'])

# Plot the results
print("Making plots...")

figs = {'training': dict(), 'test': dict()}

for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
    lqr_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                 for sim in lqr_sims[data_idx]]
    nn_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                for sim in nn_sims[data_idx]]

    figs[data_name]['cost_comparison'] = plotting.plot_total_cost(
        [sol['v'][0] for sol in data[data_idx]],
        controller_costs={'LQR': lqr_costs,
                          f'{type(nn_control).__name__:s}': nn_costs},
        title=f'Closed-loop cost evaluation ({data_name})')

    plotting.save_fig_dict(figs, config.fig_dir)

    for controller, sims in zip((lqr, nn_control), (lqr_sims, nn_sims)):
        ctrl_name = f'{type(controller).__name__:s}'
        fig_name = 'closed_loop_' + ctrl_name
        fig_dir = os.path.join(config.fig_dir, data_name, fig_name)
        plot_closed_loop(sims[data_idx], data[data_idx], t_max=config.t_int,
                         subtitle=ctrl_name + ', ' + data_name,
                         save_dir=fig_dir)

# Save data, figures, and trained controllers
data_utils.save_data(train_data, os.path.join(config.data_dir, 'train.csv'))
data_utils.save_data(test_data, os.path.join(config.data_dir, 'test.csv'))

lqr.pickle(os.path.join(config.controller_dir, 'lqr.pickle'))
nn_control.pickle(os.path.join(config.controller_dir, 'nn_control.pickle'))
