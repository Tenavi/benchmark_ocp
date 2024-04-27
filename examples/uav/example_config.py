import os

import numpy as np


# Directories where data, figures, and feedback controllers will be saved
data_dir = os.path.join('examples', 'uav', 'data')
fig_dir = os.path.join('examples', 'uav', 'figures')
controller_dir = os.path.join('examples', 'uav', 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

# Changes to default problem parameters
params = {}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 1
n_test = 1

# Distance of initial condition samples from trim
x0_distance = None

# Integration time horizon guess for infinite horizon problems
t_int = 30.

# Maximum integration time allowed
t_max = 180.

# Keyword arguments for closed-loop simulation
sim_kwargs = {'atol': 1e-08, 'rtol': 1e-04, 'ftol': 1e-05}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}

random_seed = 123

# Keyword arguments for the NN controller
nn_kwargs = {'hidden_layer_sizes': (32, 32, 32, 32), 'activation': 'tanh',
             'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-04}
