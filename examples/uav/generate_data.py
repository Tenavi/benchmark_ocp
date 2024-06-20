import argparse as ap
import os

import numpy as np

from optimalcontrol.simulate import monte_carlo
from optimalcontrol.utilities import save_data

from examples.common_utilities.supervised_learning import generate_data

from examples.uav.problem_definition import FixedWing
from examples.uav.controllers import FixedWingLQR
from examples.uav import example_config as config


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('n_traj', type=int,
                        help="Number of open loop optimal control problems to "
                             "solve. Note: slightly fewer trajectories may be "
                             "produced if the solver fails to find solutions.")
    parser.add_argument('-s', '--random_seed', type=int,
                        help="Random seed for data generation (distinct from "
                             "config.random_seed).")
    parser.add_argument('-o', '--overwrite_data', action='store_true',
                        help="If False (default), append new data to data.csv "
                             "if it already exists. If True, overwrite any "
                             "existing data.")
    args = parser.parse_args()

    n_x0 = max(1, args.n_traj)
    overwrite_data = args.overwrite_data

    ocp = FixedWing(x0_sample_seed=args.random_seed, **config.params)

    lqr = FixedWingLQR(ocp)

    x0_pool = ocp.sample_initial_conditions(n_x0, distance=config.x0_distance)
    x0_pool = x0_pool.reshape(-1, n_x0)
    
    # Warm start the optimal control solver by integrating the system with LQR
    t_eval = np.arange(0., config.t_int + config.dt_save / 2., config.dt_save)

    lqr_sims, status = monte_carlo(ocp, lqr, x0_pool, [0., config.t_int],
                                   t_eval=t_eval, **config.sim_kwargs)
    lqr_sims = np.asarray(lqr_sims, dtype=object)

    # Solve open-loop optimal control problems
    data, status, messages = generate_data(ocp, lqr_sims,
                                           **config.open_loop_kwargs)

    # Save data and LQR controller
    save_data(lqr_sims[status == 0],
              os.path.join(config.data_dir, 'LQR_sims.csv'),
              overwrite=overwrite_data)
    save_data(data[status == 0],
              os.path.join(config.data_dir, 'data.csv'),
              overwrite=overwrite_data)

    lqr.pickle(os.path.join(config.controller_dir, 'lqr.pickle'))
