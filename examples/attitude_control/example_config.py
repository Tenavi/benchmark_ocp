import os

import numpy as np


# Directories where data, figures, and feedback controllers will be saved
data_dir = os.path.join('examples', 'attitude_control', 'data')
fig_dir = os.path.join('examples', 'attitude_control', 'figures')
controller_dir = os.path.join('examples', 'attitude_control', 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

# Changes to default problem parameters
params = {}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 30
n_test = 30

# Distance in radians and radians/s (by default in l-infinity and l2 norm) of
# initial condition samples
attitude_distance = None
rate_distance = np.deg2rad(10.)

# Integration time horizon guess for infinite horizon problems
t_int = 90.

# Maximum integration time allowed
t_max = 360.

# Keyword arguments for closed-loop simulation
sim_kwargs = {'atol': 1e-08, 'rtol': 1e-04, 'method': 'RK23'}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}

random_seed = 456

# Keyword arguments for the polynomial and NN controllers
poly_kwargs = {'degree': 2, 'alpha': 100.}

nn_kwargs = {'hidden_layer_sizes': (32, 32, 32), 'activation': 'tanh',
             'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-03}

# Set True to plot Euler angles instead of quaternions in the closed-loop
# simulation plots
plot_euler = False
