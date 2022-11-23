import numpy as np

class Config:
    def __init__(
            self,
            ode_solver='RK45',
            ocp_solver='indirect',
            atol=1e-06,
            rtol=1e-03,
            fp_tol=1e-05,
            indirect_tol=1e-05,
            direct_tol=1e-06,
            direct_tol_scale=0.1,
            indirect_max_nodes=5000,
            direct_n_init_nodes=16,
            direct_n_add_nodes=16,
            direct_max_nodes=64,
            direct_max_slsqp_iter=500,
            t1_sim=60.,
            t1_max=300.,
            t1_scale=3/2,
            n_trajectories_train=100,
            n_trajectories_test=100
        ):
        '''
        Class defining (default) configuration options for setting up how ODES
        and BVPs are integrated, how many trajectories are generated and over
        what time horizons, NN architecture parameters, and training options.

        Parameters
        ----------
        ode_solver : string, default='RK45'
            ODE solver for closed loop integration. See
            scipy.integrate.solve_ivp for options.
        ocp_solver : {'indirect', 'direct'}, default='indirect'
            Whether to use an indirect method (Pontryagin's principle + boundary
            value problem solver) or direct method (pseudospectral collocation)
            to solve the open loop OCP.
        atol : float, default=1e-06
            Absolute accuracy tolerance for the ODE solver
        rtol : float, default=1e-03
            Relative accuracy tolerance for the ODE solver
        fp_tol : float, default=1e-05
            Maximum value of the vector field allowed for a trajectory to be
            considered as convergence to an equilibrium
        indirect_tol : float, default=1e-05
            Accuracy tolerance for the indirect BVP solver.
        direct_tol : float, default=1e-06
            Accuracy tolerance for the direct OCP solver.
        direct_tol_scale : float, default=0.1
            Number to multiply the accuracy tolerance for the direct OCP solver
            at each solution iteration.
        indirect_max_nodes : int, default=5000
            Maximum number of collocation points used by the indirect BVP solver.
        direct_n_init_nodes : int, default=16
            Initial number of collocation points used by the direct OCP solver.
        direct_n_add_nodes : int, default=16
            How many additional nodes to add when refining the grid used by the
            direct OCP solver.
        direct_max_nodes : int, default=64
            Maximum number of collocation points used by the direct OCP solver.
        direct_max_slsqp_iter : int, default=500
            Maximum number of iterations for the SLSQP optimization routine used
            by the direct OCP solver.
        t1_sim : float, default=60.
            Default time to integrate the ODE over
        t1_max : float, default=300.
            Maximum time horizon to integrate for.
        t1_scale : float, default=3/2
            Amount to multiply the time horizon by if need to integrate the ODE
            or BVP for longer to achieve convergence.
        n_trajectories_train : int, default=100
            Number of trajectories used for the training data set
        n_trajectories_test : int, default=100
            Number of trajectories used for the test data set
        '''
        self.ode_solver = ode_solver
        self.ocp_solver = ocp_solver

        self.atol = atol
        self.rtol = rtol
        self.fp_tol = fp_tol
        self.indirect_tol = indirect_tol
        self.direct_tol = direct_tol
        self.direct_tol_scale = direct_tol_scale

        self.indirect_max_nodes = indirect_max_nodes
        self.direct_n_init_nodes = direct_n_init_nodes
        self.direct_n_add_nodes = direct_n_add_nodes
        self.direct_max_nodes = direct_max_nodes
        self.direct_max_slsqp_iter = direct_max_slsqp_iter

        self.t1_sim = t1_sim
        self.t1_scale = np.maximum(t1_scale, 1. + 1e-02)
        self.t1_max = np.maximum(t1_max, t1_sim * self.t1_scale)

        self.n_trajectories_train = n_trajectories_train
        self.n_trajectories_test = n_trajectories_test
