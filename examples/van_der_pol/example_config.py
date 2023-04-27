# Number of training and test trajectories
n_train = 100
n_test = 100

# Distance (by default in l-infinity norm) of initial condition samples from xf
x0_distance = 3.0

# Integration time horizon guess for infinite horizon problems
t_int = 10. * x0_distance

# Maximum integration time allowed
t_max = 5. * t_int

# Changes to default problem parameters
params = {}

# Keyword arguments for closed-loop simulation
sim_kwargs = {}

# Keyword arguments for open-loop data generation
open_loop_kwargs = {}
