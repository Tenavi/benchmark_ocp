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

# ------------------------------------------------------------------------------

class StateSampler:
    '''Generic base class for algorithms to sample states.'''
    def __init__(self, *args, **kwargs):
        pass

    def update(self, **kwargs):
        '''Update parameters of the sampler.'''
        pass

    def __call__(self, n_samples=1, **kwargs):
        '''
        Generate samples of the system state.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        kwargs
            Other keyword arguments implemented by the subclass.

        Returns
        -------
        x : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x` will be a one-dimensional array.
        '''
        raise NotImplementedError

class UniformSampler(StateSampler):
    '''Class which implements uniform sampling from a hypercube.'''
    def __init__(self, lb, ub, xf, norm=2, seed=None):
        '''
        Parameters
        ----------
        lb : (n_states,) or (n_states,1) array
            Lower bounds for each dimension of the hypercube.
        ub : (n_states,) or (n_states,1) array
            Upper bounds for each dimension of the hypercube.
        xf : (n_states,) or (n_states,1) array
            Nominal state within the hypercube. If `sample` is called with a
            specified `distance` argument, this distance is calculated from `xf`
            with norm specified by `norm`.
        norm : {1,2}, default=2
            Order of the norm (l1 or l2) with which to calculate distances from
            `xf`.
        seed : int, optional
            Random seed for the random number generator.
        '''
        self.update(lb=lb, ub=ub, xf=xf)

        if norm not in [1,2]:
            raise ValueError('norm must be 1 or 2')
        self.norm = int(norm)

        self.rng = np.random.default_rng(seed)

    def update(self, lb=None, ub=None, xf=None, seed=None):
        '''
        Update parameters of the sampler.

        Parameters
        '''
        if lb is not None:
            self.lb = np.reshape(lb, (-1,1))
        if ub is not None:
            self.ub = np.reshape(ub, (-1,1))
        if xf is not None:
            self.xf = np.reshape(xf, (-1,1))
            self.n_states = self.xf.shape[0]

        if self.n_states != self.ub.shape[0] or self.n_states != self.lb.shape[0]:
            raise ValueError('lb, ub, and xf must have compatible shapes.')

        if not np.all(self.xf <= self.ub) or not np.all(self.lb <= self.xf):
            raise ValueError('Must have lb <= xf <= ub.')

        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, n_samples=1, distance=None):
        '''
        Generate samples of the system state uniformly in a hypercube with lower
        and upper bound specified by `self.lb` and `self.ub`, respectively.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired distance (in l1 or l2 norm) of samples from `self.xf`. The
            type of norm is determined by `self.norm`. Note that depending on
            how `distance` is specified, samples may be outside the hypercube.

        Returns
        -------
        x : (n_states, n_samples) or (n_states,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples=1` then `x` will be a one-dimensional array.
        '''
        if not n_samples:
            raise ValueError('n_samples must be a positive int.')

        x = self.rng.uniform(
            low=self.lb, high=self.ub, size=(self.n_states, n_samples)
        )

        if distance is not None:
            x -= self.xf
            x_norm = distance / np.linalg.norm(x, self.norm, axis=0)
            x = x_norm * x + self.xf

        if n_samples == 1:
            return x.flatten()
        return x

# ------------------------------------------------------------------------------

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
    raise NotImplementedError

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
