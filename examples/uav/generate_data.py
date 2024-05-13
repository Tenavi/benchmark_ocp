import os
import time

import numpy as np
from tqdm import tqdm

from optimalcontrol.controls import LinearQuadraticRegulator

from examples.common_utilities import data_utils

from examples.uav.problem_definition import FixedWing
from examples.uav import example_config as config
from examples.uav.plot_fixed_wing import plot_fixed_wing


if __name__ == '__main__':
    # Initialize the optimal control problem
    random_seed = getattr(config, 'random_seed', None)
    if random_seed is None:
        random_seed = int(time.time())

    ocp = FixedWing(x0_sample_seed=random_seed, **config.params)

    xf = ocp.parameters.trim_state.to_array()
    uf = ocp.parameters.trim_controls.to_array()
    
    # Create an LQR controller as a baseline
    # System matrices (vector field Jacobians)
    A, B = ocp.jac(xf, uf)
    # Cost matrices (1/2 Running cost Hessians)
    Q, R = ocp.running_cost_hess(xf, uf)

    lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R,
                                   xf=ocp.trim_state, uf=ocp.trim_controls,
                                   u_lb=ocp.control_lb, u_ub=ocp.control_ub)

    # Generate some training and test data
    
    # First sample initial conditions
    x0_pool = ocp.sample_initial_conditions(config.n_train + config.n_test,
                                            distance=config.x0_distance)
    
    # Warm start the optimal control solver by integrating the system with LQR
    t = np.arange(0., config.t_int + config.dt / 2., config.dt)
    n_t = t.shape[-1]
    downsample_idx = np.arange(0, n_t + 1, 10)

    lqr_sims = []

    for x0 in x0_pool.T:
        sim = {'t': t[downsample_idx],
               'x': np.empty((ocp.n_states, n_t)),
               'u': np.empty((ocp.n_controls, n_t))}

        sim['x'][:, 0] = x0

        for k in tqdm(range(n_t)):
            # Rescale altitude so LQR doesn't act badly for large commands
            x_lqr = np.copy(sim['x'][:, k])
            x_lqr[0] = ocp.parameters.h_cost_ceil * ocp.scale_altitude(x_lqr[0])
            # Compute LQR control
            sim['u'][:, k] = lqr(x_lqr)

            if k < n_t - 1:
                # Euler forward integration to save time
                f = ocp.dynamics(sim['x'][:, k], sim['u'][:, k])
                sim['x'][:, k + 1] = sim['x'][:, k] + config.dt * f

        # Downsample states and controls
        for arg in ['x', 'u']:
            sim[arg] = sim[arg][:, downsample_idx]

        sim['v'] = ocp.total_cost(sim['t'], sim['x'], sim['u'])[::-1]
        sim['L'] = ocp.running_cost(sim['x'], sim['u'])

        lqr_sims.append(sim)

    # Solve open loop optimal control problems
    data, status, messages = data_utils.generate_data(ocp, lqr_sims,
                                                      **config.open_loop_kwargs)
    
    print("\n" + "+" * 80 + "\n")

    # Save data and LQR controller
    data_utils.save_data(lqr_sims,
                         os.path.join(config.data_dir, 'lqr_sims.csv'))
    data_utils.save_data(data, os.path.join(config.data_dir, 'data.csv'))

    lqr.pickle(os.path.join(config.controller_dir, 'lqr.pickle'))
    
    # Plot the results
    print("Making plots...")

    plot_fixed_wing(ocp, data, save_dir=os.path.join(config.fig_dir, 'data'))

    for i in range(len(lqr_sims)):
        fig_dir = os.path.join(config.fig_dir, f'lqr_sim_{i}')
        plot_fixed_wing(ocp, [lqr_sims[i], data[i]],
                        sim_labels=['LQR', 'optimal'], save_dir=fig_dir)
