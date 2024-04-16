import os
import time

import numpy as np
from scipy.interpolate import make_interp_spline

from optimalcontrol import simulate, utilities, analyze
from optimalcontrol.controls import LinearQuadraticRegulator
from optimalcontrol.open_loop import solve_infinite_horizon

from examples.common_utilities import data_utils, plotting
from examples.common_utilities.dynamics import quaternion_to_euler, euler_to_quaternion

from examples.uav.problem_definition import FixedWing
from examples.uav import example_config as config


if __name__ == '__main__':
    # Initialize the optimal control problem
    random_seed = getattr(config, 'random_seed', None)
    if random_seed is None:
        random_seed = int(time.time())
    rng = np.random.default_rng(random_seed + 1)

    ocp = FixedWing(x0_sample_seed=random_seed, **config.params)

    xf = ocp.parameters.trim_state.to_array()
    uf = ocp.parameters.trim_controls.to_array()
    
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

    """
    
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
    
    print("\n" + "+" * 80 + "\n")
    
    for sim in lqr_sims:
        sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]
        sim['L'] = ocp.running_cost(sim['x'], sim['u'])
    
    # Plot the results
    print("Making plots...")
    
    figs = {'training': dict(), 'test': dict()}
    
    for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
        lqr_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                     for sim in lqr_sims[data_idx]]
    
        plotting.save_fig_dict(figs, config.fig_dir)
    
        ctrl_name = f'{type(lqr).__name__:s}'
        fig_name = 'closed_loop_' + ctrl_name
        fig_dir = os.path.join(config.fig_dir, data_name, fig_name)
        plot_closed_loop(lqr_sims[data_idx], data[data_idx], t_max=config.t_int,
                         subtitle=ctrl_name + ', ' + data_name,
                         save_dir=fig_dir)
    
    # Save data, figures, and trained controllers
    data_utils.save_data(train_data, os.path.join(config.data_dir, 'train.csv'))
    data_utils.save_data(test_data, os.path.join(config.data_dir, 'test.csv'))
    
    lqr.pickle(os.path.join(config.controller_dir, 'lqr.pickle'))
    """
