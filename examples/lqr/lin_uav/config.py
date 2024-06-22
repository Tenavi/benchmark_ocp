""""Config file for linearized longitudinal dynamics of a fixed wing UAV as an LQR problem."""

import os
import numpy as np

# Directories where data, figures, and feedback controllers will be saved
main_dir = os.path.join('examples', 'lqr', 'lin_uav')
data_dir = os.path.join(main_dir, 'data')
fig_dir = os.path.join(main_dir, 'figures')
controller_dir = os.path.join(main_dir, 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

random_seed = 123
problem_name = "Linearized UAV's Longitudinal Dynamics"

# Problem definition definition
# see lecture matetrial chap5.pdf in https://github.com/randybeard/uavbook and the corresponding uavbook.pdf
xf = np.array([19.8961, 2.0357, 0, 0.1020, 0])
uf = np.array([-0.2624, 0.5210])
A = np.array([[-0.2529, 0.6962, -1.9870, -9.7491, 0], 
              [-0.6108, -3.6183, 19.4199, -0.9979, 0],
              [0.3036, -2.9669, -4.2358, 0, 0],
              [0, 0, 1, 0, 0],
              [0.1018, -0.9948, 0, 20, 0]])
B = np.array([[-0.0025, 5.3843], [-1.6575, 0], [-23.1119, 0], [0, 0], [0, 0]])
Q = np.diag([1/40, 1, (6/np.pi)**2, 1, 1/1000])
R = np.diag([1/625, 0.1])

x0_lb = xf - np.array([5, 5, np.pi/6, np.pi/3, 10])
x0_ub = xf + np.array([5, 5, np.pi/6, np.pi/3, 10])
lqr_param_dict = {
    'A': A, 'B': B, 'Q': Q, 'R': R,
    'u_lb': np.array([-0.44, 0]), 'u_ub': np.array([0.44, 1]), 'xf': xf, 'uf': uf
}
x0_bounds = {'x0_lb': x0_lb, 'x0_ub': x0_ub}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 25
n_test = 25

# Distance (by default in l-infinity norm) of initial condition samples
x0_distance = None

# Integration time horizon guess for infinite horizon problems
t_int = 20

# Maximum integration time allowed
t_max = 5. * t_int

# Keyword arguments for closed-loop simulation
sim_kwargs = {'atol': 1e-08, 'rtol': 1e-04, 'method': 'BDF'}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}

# Keyword arguments for the NN controller
controller_kwargs = {'hidden_layer_sizes': (32, 32), 'activation': 'tanh',
                     'solver': 'lbfgs', 'max_iter': 10000, 'tol': 1e-03}
