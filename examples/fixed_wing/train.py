import os
import time

import numpy as np

from optimalcontrol import analyze
from optimalcontrol.utilities import load_data, stack_dataframes

from examples.common_utilities.supervised_learning import (
    QuaternionControlWrapper, SimpleQRnet, NeuralNetworkController,
    RandomFourierController)

from examples.fixed_wing.problem_definition import FixedWing
from examples.fixed_wing.controllers import FixedWingLQR
from examples.fixed_wing import example_config as config


if __name__ == '__main__':
    # Initialize the optimal control problem
    random_seed = getattr(config, 'random_seed', None)
    if random_seed is None:
        random_seed = int(time.time())
    rng = np.random.default_rng(random_seed)

    ocp = FixedWing(**config.params)

    q0_state = ocp.n_states - 1

    lqr = FixedWingLQR(ocp)

    # Load the training and test datasets
    train_data = load_data(os.path.join(config.data_dir, 'train_data.csv'))
    test_data = load_data(os.path.join(config.data_dir, 'test_data.csv'))

    # Extract the number of desired trajectories
    train_idx = np.arange(len(train_data))
    rng.shuffle(train_idx)
    train_data = train_data[train_idx[:config.n_train]]

    # Turn data into numpy arrays
    _, x_train, u_train, _, _ = stack_dataframes(*train_data)
    _, x_test, u_test, _, _ = stack_dataframes(*test_data)

    controllers = [lqr,
                   NeuralNetworkController(
                       random_state=random_seed + 1, **config.nn_kwargs),
                   SimpleQRnet(lqr, NeuralNetworkController(
                       random_state=random_seed + 1, **config.nn_kwargs)),
                   SimpleQRnet(lqr, RandomFourierController(
                       random_state=random_seed + 2, **config.rff_kwargs))]

    for i in range(1, len(controllers)):
        controllers[i] = QuaternionControlWrapper(q0_state, controllers[i])
        print(f"\nTraining {controllers[i]}...")
        controllers[i].train(x_train, u_train)

    print("\n" + "+" * 80)

    for controller in controllers:
        print(f"\nLinear stability analysis for {controller}:")

        x, status = analyze.find_equilibrium(ocp, controller, lqr.xf,
                                             2. * config.t_int, config.t_max,
                                             **config.sim_kwargs)
        if np.any(status == 0):
            stability = f"{'un' if status[1] == 0 else ''}stable"
            print(f"Found likely {stability} equilibrium:")
            print(x[:, status == 0].reshape(-1, 1))
            analyze.linear_stability(ocp, controller, x[:, status == 0],
                                     zero_tol=1e-05)
        else:
            print("No equilibrium found. Forward integration ended at")
            print(x[:, :1])
            print(f"(forward/backward status = {status})")

    print("\n" + "+" * 80)

    for controller in controllers:
        # Report approximation accuracy
        train_r2 = controller.r2_score(x_train, u_train)
        test_r2 = controller.r2_score(x_test, u_test)
        print(f"\n{controller} R2 score: {train_r2:.4f} (train), "
              f"{test_r2:.4f} (test)")

        # Save trained controllers
        controller._train_idx = train_idx
        controller.pickle(os.path.join(config.controller_dir,
                                       f'{controller}.pickle'))
