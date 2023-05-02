# Changes to default problem parameters
params = {}

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

random_seed = 123

# Keyword arguments for the SVR controller
controller_kwargs = {'hidden_layer_sizes': (32, 32, 32), 'activation': 'tanh',
                     'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-03}
