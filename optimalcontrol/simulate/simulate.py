import numpy as np

from .ivp import solve_ivp

def integrate_closed_loop(
        problem, controller, tspan, x0, t_eval=None,
        method="RK45", atol=1e-06, rtol=1e-03
    ):
    """
    Integrate system dynamics with a given feedback controller for a fixed time
    horizon.

    Parameters
    ----------
    problem
    controller
    tspan
    x0
    t_eval
    method
    atol
    rtol

    Returns
    -------
    t
    x
    status
    """
    def fun(t, x):
        return problem.dynamics(x, controller(x))

    def jac(t, x):
        return problem.jacobian(x, controller)

    ode_sol = solve_ivp(
        fun, tspan, x0, jac=jac, events=problem.integration_events,
        t_eval=t_eval, vectorized=True, method=method, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def integrate_to_converge(dynamics, jacobian, controller, X0, config, events=None):
    '''
    Simulate the closed-loop system until reach t_max or the dX/dt = 0.

    Parameters
    ----------
        OCP: instance of a setupProblem class defining dynamics, Jacobian, etc.
        config: a configuration dict defined in problem_def.py
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
        converged: whether or not equilibrium was reached, bool
    '''

    t = np.zeros(1)
    X = X0.reshape(-1,1)

    converged = False

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while not converged and t[-1] < config.t1_max:
        t1 = np.maximum(config.t1_sim, t[-1] * config.t1_scale)
        # Simulate the closed-loop system
        t_new, X_new, status = integrate_closed_loop(
            dynamics,
            jacobian,
            controller,
            [t[-1], t1],
            X[:,-1],
            events=events,
            method=config.ode_solver,
            atol=config.atol,
            rtol=config.rtol
        )

        t = np.concatenate((t, t_new[1:]))
        X = np.hstack((X, X_new[:,1:]))

        if status == 1:
            break

        U = controller.eval_U(X[:,-1])
        converged = np.linalg.norm(dynamics(X[:,-1], U)) < config.fp_tol

    return t, X, converged
