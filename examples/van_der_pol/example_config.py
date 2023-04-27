# Number of training and test trajectories
n_train = 10
n_test = 10

# Distance (by default in l-infinity norm) of initial condition samples from xf
x0_distance = 1.0

# Integration time horizon guess for infinite horizon problems
t_int = 10.

# Maximum integration time allowed
t_max = 100.

# Changes to default problem parameters
params = {}

# Keyword arguments for closed-loop simulation
sim_kwargs = {}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {'verbose': 0}
