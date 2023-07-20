import os


# Directories where data, figures, and feedback controllers will be saved
data_dir = os.path.join('examples', 'burgers', 'data')
fig_dir = os.path.join('examples', 'burgers', 'figures')
controller_dir = os.path.join('examples', 'burgers', 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

# Changes to default problem parameters
params = {}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 30
n_test = 20

# Quadrature integrated norm of initial condition samples (leave as None for
# uniform sampling)
x0_distance = None

# Integration time horizon guess for infinite horizon problems
t_int = 20.

# Maximum integration time allowed
t_max = 120.

# Keyword arguments for closed-loop simulation
sim_kwargs = {'method': 'LSODA'}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {'tol': 1e-04, 'max_nodes': 2000}

random_seed = 123

# Keyword arguments for the NN controller
nn_kwargs = {'hidden_layer_sizes': (32, 32, 32, 32), 'activation': 'tanh',
             'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-04}
