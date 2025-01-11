import os


# Directories where data, figures, and feedback controllers will be saved
data_dir = os.path.join('examples', 'fixed_wing', 'data')
fig_dir = os.path.join('examples', 'fixed_wing', 'figures')
controller_dir = os.path.join('examples', 'fixed_wing', 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

# Changes to default problem parameters
params = {}

# Number of training trajectories
n_train = 400

# Distance of initial condition samples from trim
x0_distance = None

# Integration time horizon guess for infinite horizon problems
t_int = 30.

# Maximum time horizon allowed for closed-loop integration
t_max = 120.

# Temporal resolution for saving closed loop simulation data
dt_save = 0.1

# Keyword arguments for closed-loop simulation
sim_kwargs = {'method': 'RK23'}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {'method': 'direct', 'time_scale': 0.5,
                    'ivp_options': sim_kwargs}

# Keyword arguments for the K-nearest neighbors and NN controllers
k_nn_kwargs = {'weights': 'distance', 'n_neighbors': 100}

nn_kwargs = {'hidden_layer_sizes': (32, 32, 32), 'activation': 'tanh',
             'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-04}

random_seed = None
