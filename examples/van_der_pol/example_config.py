# Number of training and test trajectories
n_train = 50
n_test = 50

# Distance (by default in l-infinity norm) of initial condition samples
x0_distance = 3.0

# Integration time horizon guess for infinite horizon problems
t_int = 10. * x0_distance

# Maximum integration time allowed
t_max = 5. * t_int

# Changes to default problem parameters
params = {'u_lb': -0.8, 'u_ub': 0.8}

# Keyword arguments for closed-loop simulation
sim_kwargs = {}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}

# Keyword arguments for the SVR controller
controller_kwargs = {'tol': 1e-06, 'C': 5.0, 'verbose': True}
