import numpy as np

def saturate(u, u_lb, u_ub):
    '''
    Hard saturation of control for numpy arrays.

    Parameters
    ----------
    u : (n_controls, n_data) or (n_controls,) array
        Control(s) to saturate.
    u_lb : (n_controls, 1) array
        Lower control bounds.
    u_ub : (n_controls, 1) array
        upper control bounds.

    Returns
    -------
    u : array with same shape as input
        Control(s) saturated between u_lb and u_ub
    '''
    if u_lb is not None or u_ub is not None:
        if u.ndim < 2:
            u = np.clip(u, u_lb.flatten(), u_ub.flatten())
        else:
            u = np.clip(u, u_lb, u_ub)

    return u

def find_fixed_point(OCP, controller, tol, X0=None, verbose=True):
    '''
    Use root-finding to find a fixed point (equilibrium) of the closed-loop
    dynamics near the desired goal state OCP.X_bar. ALso computes the
    closed-loop Jacobian and its eigenvalues.

    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    tol : float
        Maximum value of the vector field allowed for a trajectory to be
        considered as convergence to an equilibrium
    X0 : array, optional
        Initial guess for the fixed point. If X0=None, use OCP.X_bar
    verbose : bool, default=True
        Set to True to print out the deviation of the fixed point from OCP.X_bar
        and the Jacobian eigenvalue

    Returns
    -------
    X_star : (n_states, 1) array
        Closed-loop equilibrium
    X_star_err : float
        ||X_star - OCP.X_bar||
    F_star : (n_states, 1) array
        Vector field evaluated at X_star. If successful should have F_star ~ 0
    Jac : (n_states, n_states) array
        Close-loop Jacobian at X_star
    eigs : (n_states, 1) complex array
        Eigenvalues of the closed-loop Jacobian
    max_eig : complex scalar
        Largest eigenvalue of the closed-loop Jacobian
    '''
    if X0 is None:
        X0 = OCP.X_bar
    X0 = np.reshape(X0, (OCP.n_states,))

    def dynamics_wrapper(X):
        U = controller.eval_U(X)
        F = OCP.dynamics(X, U)
        C = OCP.constraint_fun(X)
        if C is not None:
            F = np.concatenate((F.flatten(), C.flatten()))
        return F

    def Jacobian_wrapper(X):
        J = OCP.closed_loop_jacobian(X, controller)
        JC = OCP.constraint_jacobian(X)
        if JC is not None:
            J = np.vstack((
                J.reshape(-1,X.shape[0]), JC.reshape(-1,X.shape[0])
            ))
        return J

    sol = root(dynamics_wrapper, X0, jac=Jacobian_wrapper, method='lm')

    X_star = sol.x.reshape(-1,1)
    U_star = controller(X_star)
    F_star = OCP.dynamics(X_star, U_star).reshape(-1,1)
    Jac = OCP.closed_loop_jacobian(sol.x, controller)

    X_star_err = OCP.norm(X_star)[0]

    eigs = np.linalg.eigvals(Jac)
    idx = np.argsort(eigs.real)
    eigs = eigs[idx].reshape(-1,1)
    max_eig = np.squeeze(eigs[-1])

    # Some linearized systems always have one or more zero eigenvalues.
    # Handle this situation by taking the next largest.
    if np.abs(max_eig.real) < tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < tol**2,
                np.abs(max_eig0.real) < tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    if verbose:
        s = '||actual - desired_equilibrium|| = {norm:1.2e}'
        print(s.format(norm=X_star_err))
        if np.max(np.abs(F_star)) > tol:
            print('Dynamics f(X_star):')
            print(F_star)
        s = 'Largest Jacobian eigenvalue = {real:1.2e} + j{imag:1.2e} \n'
        print(s.format(real=max_eig.real, imag=np.abs(max_eig.imag)))

    return X_star, X_star_err, F_star, Jac, eigs, max_eig

def cross_product_matrix(w):
    zeros = np.zeros_like(w[0])
    wx = np.array([
        [zeros, -w[2], w[1]],
        [w[2], zeros, -w[0]],
        [-w[1], w[0], zeros]]
    )
    return wx
