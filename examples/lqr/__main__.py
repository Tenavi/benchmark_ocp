import os
import time
import argparse as ap

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from optimalcontrol import controls, simulate, utilities, analysis
from optimalcontrol.problem.linear_quadratic import LinearQuadraticProblem

from examples.lqr import lin_vdp_config as config
from examples.common_utilities import data_utils, supervised_learning, plotting

parser = ap.ArgumentParser()
parser.add_argument('-p', '--problem', default='double_int',
                    help="Specify LQR problem to get config")
args = parser.parse_args()


if args.problem == 'double_int':
    from examples.lqr import double_int_config as config
elif args.problem == 'lin_vdp':
    from examples.lqr import lin_vdp_config as config
else:
    raise NotImplementedError("The config file is not defined for problem: {}".format(args.problem))

print("\nSolving LQR problem: {}".format(config.problem_name))
print("\n" + "+" * 80 + "\n")

random_seed = getattr(config, 'random_seed', None)
if random_seed is None:
    random_seed = int(time.time())
rng = np.random.default_rng(random_seed + 1)


ocp = LinearQuadraticProblem(**config.lqr_param_dict, **config.x0_bounds)
lqr = controls.LinearQuadraticRegulator(**config.lqr_param_dict)

# First sample initial conditions
x0_pool = ocp.sample_initial_conditions(config.n_train + config.n_test,
                                        distance=config.x0_distance)

lqr_sims, status = simulate.monte_carlo_to_converge(
    ocp, lqr, x0_pool, config.t_int, config.t_max, **config.sim_kwargs)
lqr_data = np.asarray(lqr_sims, dtype=object)

# Reserve a subset of data for training and use the rest for testing
data_idx = np.arange(x0_pool.shape[1])[status == 0]
rng.shuffle(data_idx)
test_idx = data_idx[config.n_train:]
train_idx = data_idx[:config.n_train]

train_data = lqr_data[train_idx]
test_data = lqr_data[test_idx]

# Turn data into numpy arrays for training and test evaluation
_, x_train, u_train, _, _ = utilities.stack_dataframes(*train_data)
_, x_test, u_test, _, _ = utilities.stack_dataframes(*test_data)

print("\nTraining neural network controller...")
controller = supervised_learning.NeuralNetworkController(
    x_train, u_train,
    u_lb=ocp.parameters.u_lb,
    u_ub=ocp.parameters.u_ub,
    random_state=random_seed + 2,
    **config.controller_kwargs
)

print(f"\nLinear stability analysis for {type(controller).__name__:s}:")

x, f = analysis.find_equilibrium(ocp, controller, config.lqr_param_dict['xf'])
jac = utilities.closed_loop_jacobian(x, ocp.jac, controller)
eigs, max_eig = analysis.linear_stability(jac)

# If an unstable equilibrium was found, try perturbing the equilibrium
# slightly and integrating from there to find a stable equilibrium
if max_eig > 0.:
    x += rng.normal(scale=1/100, size=x.shape)
    t, x, _ = simulate.integrate_to_converge(
        ocp, controller, x, config.t_int, config.t_max, **config.sim_kwargs)

    print("Closed-loop integration from the unstable equilibrium led to:")
    print(x[:, -1:])
    print("Retrying root-finding...")

    x, f = analysis.find_equilibrium(ocp, controller, x[:, -1])
    jac = utilities.closed_loop_jacobian(x, ocp.jac, controller)
    eigs, max_eig = analysis.linear_stability(jac)

print("\n" + "+" * 80)

train_r2 = r2_score(u_train.T, controller(x_train).T)
test_r2 = r2_score(u_test.T, controller(x_test).T)
print(f"\n{type(controller).__name__:s} R2 score: {train_r2:.4f} (train) "
          f"{test_r2:.4f} (test)")

print("\n" + "+" * 80 + "\n")

# Evaluate performance of the NN controller in closed-loop simulation
sims, status = simulate.monte_carlo_to_converge(
    ocp, controller, x0_pool, config.t_int, config.t_max, **config.sim_kwargs)

for sim in sims:
    sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]
    sim['L'] = ocp.running_cost(sim['x'], sim['u'])
for sim in lqr_data:
    sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]

# Plot the results
figs = {'training': dict(), 'test': dict()}
for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
    nn_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                for sim in sims[data_idx]]
    figs[data_name]['cost_comparison'] = plotting.plot_total_cost(
        [sol['v'][0] for sol in lqr_data[data_idx]],
        controller_costs={f'{type(controller).__name__:s}': nn_costs},
        title=f'Closed-loop cost evaluation ({data_name})')
    
    ctrl_name = f'{type(controller).__name__:s}'
    figs[data_name]['closed_loop_' + ctrl_name] = plotting.plot_closed_loop(
        sims[data_idx], t_max=config.t_int,
        subtitle=ctrl_name + ', ' + data_name)
    
    figs[data_name]['closed_loop_3d' + ctrl_name] = plotting.plot_closed_loop_3d(
        sims[data_idx], lqr_data[data_idx], controller_name=ctrl_name,
        title=f'Closed-loop trajectories and controls ({ctrl_name}, 'f'{data_name})')

# Save data, figures, and trained NN
data_utils.save_data(train_data, os.path.join(config.data_dir, 'train.csv'))
data_utils.save_data(test_data, os.path.join(config.data_dir, 'test.csv'))

for data_name, figs_subset in figs.items():
    _fig_dir = os.path.join(config.fig_dir, data_name)
    os.makedirs(_fig_dir, exist_ok=True)
    for fig_name, fig in figs_subset.items():
        plt.figure(fig)
        plt.savefig(os.path.join(_fig_dir, fig_name + '.pdf'))

controller.pickle(os.path.join(config.controller_dir, 'nn_control.pickle'))
