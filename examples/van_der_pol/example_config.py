import os


# Directories where data, figures, and feedback controllers will be saved
main_dir = os.path.join('examples', 'van_der_pol')
data_dir = os.path.join(main_dir, 'data')
fig_dir = os.path.join(main_dir, 'figures')
controller_dir = os.path.join(main_dir, 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

# Changes to default problem parameters
params = {}

# Number of training and test trajectories
# Note: slightly fewer training trajectories may be produced if the solver fails
# to find open-loop solutions
n_train = 40
n_test = 40

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

# Keyword arguments for the supervised learning controllers
k_nn_kwargs = {'weights': 'distance', 'n_neighbors': 10}

nn_kwargs = {'hidden_layer_sizes': (32, 32, 32), 'activation': 'tanh',
             'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-03}

rff_kwargs = {'sigma': [2., 1.], 'n_features': 200}

random_seed = 1234
