""""Config file for double integrator as an LQR problem."""

import os
import numpy as np

# Directories where data, figures, and feedback controllers will be saved
main_dir = os.path.join('examples', 'lqr', 'double_int')
data_dir = os.path.join(main_dir, 'data')
fig_dir = os.path.join(main_dir, 'figures')
controller_dir = os.path.join(main_dir, 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

random_seed = 123
problem_name = "Double Integrator"

# Double integrator problem definition
# see https://underactuated.mit.edu/dp.html#example1 and 
# https://underactuated.mit.edu/lqr.html#example1 for more details
A = np.array([[0., 1.], [0., 0.]])
B = np.array([[0.], [1.]])
Q = np.eye(2)
R = np.eye(1)
xf = np.array([0., 0.])
uf = 0.
x0_lb = -np.array([[3.], [3.]])
x0_ub = np.array([[3.], [3.]])
lqr_param_dict = {
    'A': A, 'B': B, 'Q': Q, 'R': R, 'u_lb': None, 'u_ub': None, 'xf': xf, 'uf': uf
}
x0_bounds = {'x0_lb': x0_lb, 'x0_ub': x0_ub}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 25
n_test = 25

# Distance (by default in l-infinity norm) of initial condition samples
x0_distance = 3.0

# Integration time horizon guess for infinite horizon problems
t_int = 10. * x0_distance

# Maximum integration time allowed
t_max = 5. * t_int

# Keyword arguments for closed-loop simulation
sim_kwargs = {'atol': 1e-08, 'rtol': 1e-04, 'method': 'RK23'}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}

# Keyword arguments for the NN controller
controller_kwargs = {'hidden_layer_sizes': (32, 32), 'activation': 'relu',
                     'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-03}
