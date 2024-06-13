import argparse as ap
import os
import time

import numpy as np
from tqdm import tqdm

from optimalcontrol.controls import LinearQuadraticRegulator

from examples.common_utilities import data_utils

from examples.uav.problem_definition import FixedWing
from examples.uav import example_config as config


if __name__ == '__main__':
    parser = ap.ArgumentParser()

    parser.add_argument("n_traj", type=int,
                        help="Number of open loop optimal control problems to "
                             "solve. Note: slightly fewer trajectories may be "
                             "produced if the solver fails to find solutions.")
    parser.add_argument("-o", "--overwrite_data", action='store_true',
                        help="If False (default), append new data to data.csv "
                             "if it already exists. If True, overwrite any "
                             "existing data.")
    args = parser.parse_args()

    n_x0 = max(1, args.n_traj)
    overwrite_data = args.overwrite_data

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

    x0_pool = ocp.sample_initial_conditions(n_x0, distance=config.x0_distance)
    x0_pool = x0_pool.reshape(-1, n_x0)
    
    # Warm start the optimal control solver by integrating the system with LQR
    t = np.arange(0., config.t_int + config.dt / 2., config.dt)
    n_t = t.shape[-1]
    downsample_idx = np.arange(0, n_t + 1, 10)

    lqr_sims = []

    print(f"Simulating LQR-controlled system to generate initial guesses...")

    for i in tqdm(range(x0_pool.shape[1])):
        sim = {'t': t[downsample_idx],
               'x': np.empty((ocp.n_states, n_t)),
               'u': np.empty((ocp.n_controls, n_t))}

        sim['x'][:, 0] = x0_pool[:, i]

        for k in range(n_t):
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

        lqr_sims.append(sim)

    lqr_sims = np.array(lqr_sims, dtype=object)

    # Solve open loop optimal control problems
    data, status, messages = data_utils.generate_data(ocp, lqr_sims,
                                                      **config.open_loop_kwargs)

    # Save data and LQR controller
    data_utils.save_data(lqr_sims[status == 0],
                         os.path.join(config.data_dir, 'LQR_sims.csv'),
                         overwrite=overwrite_data)
    data_utils.save_data(data[status == 0],
                         os.path.join(config.data_dir, 'data.csv'),
                         overwrite=overwrite_data)

    lqr.pickle(os.path.join(config.controller_dir, 'lqr.pickle'))
