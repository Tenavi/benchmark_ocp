""""Config file for linearized Van der Pol oscillator as an LQR problem."""

import os
from examples.van_der_pol import VanDerPol

# Directories where data, figures, and feedback controllers will be saved
data_dir = os.path.join('examples', 'lqr', 'lin_vdp', 'data')
fig_dir = os.path.join('examples', 'lqr', 'lin_vdp', 'figures')
controller_dir = os.path.join('examples', 'lqr', 'lin_vdp', 'controllers')

for directory in [data_dir, fig_dir, controller_dir]:
    os.makedirs(directory, exist_ok=True)

random_seed = 123
problem_name = "Linerized Van der Pol oscillator"

# Changes to default problem parameters
params = {}

# Create the LQR problem by linearizing Van-der-pol example at the terminal state
vdp = VanDerPol(x0_sample_seed=random_seed, **params)
xf = vdp.parameters.xf.flatten()
uf = vdp.parameters.uf
# System matrices (vector field Jacobians)
A, B = vdp.jac(xf, uf)
# Cost matrices (1/2 Running cost Hessians)
Q, R = vdp.running_cost_hess(xf, uf)
lqr_param_dict = {
    'A': A, 'B': B, 'Q': Q, 'R': R, 'u_lb': None, 'u_ub': None, 'xf': xf, 'uf': uf
}
x0_bounds = {'x0_lb': vdp.parameters.x0_lb, 'x0_ub': vdp.parameters.x0_ub}

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

# Keyword arguments for the SVR controller
controller_kwargs = {'hidden_layer_sizes': (32, 32), 'activation': 'relu',
                     'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-03}
