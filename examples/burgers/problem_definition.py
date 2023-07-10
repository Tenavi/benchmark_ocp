import numpy as np
from scipy.spatial.distance import cdist

from optimalcontrol.problem import OptimalControlProblem


class BurgersPDE(OptimalControlProblem):
    _required_parameters = {'n_states': 64, 'nu': 0.2, 'gamma': 0.1, 'R': 4.,
                            'kappa': 25.}
    _optional_parameters = {'u_lb': -1., 'u_ub': 1., 'x0_sample_seed': None}

    @property
    def n_states(self):
        return self.parameters.n_states

    @property
    def n_controls(self):
        return 2

    @property
    def final_time(self):
        return np.inf

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'n_states' in new_params:
            n_states = new_params['n_states']
            if not isinstance(n_states, int):
                raise TypeError(f"type(n_states) is {type(n_states):s} but "
                                f"must be an int")

            if hasattr(obj, '_D') and obj._D.shape[0] != n_states:
                raise UserWarning(f"Changing n_states to {n_states:d}")
                
            # Generate Chebyshev nodes, differentiation matrices, and
            # Clenshaw-Curtis weights
            obj._xi, obj._D, obj._w = cheb(int(n_states) + 1)
            obj._D2 = np.matmul(obj._D, obj._D)

            # Truncates system to account for zero boundary conditions
            obj._xi = obj._xi[1:-1].reshape(-1, 1)
            obj._w = obj._w[1:-1]
            obj._D = obj._D[1:-1, 1:-1]
            obj._D2 = obj._D2[1:-1, 1:-1]

        if 'n_states' in new_params or 'kappa' in new_params:
            # Control effectiveness matrix
            obj.kappa = float(obj.kappa)

            B = obj.kappa * np.hstack(((obj._xi + 4 / 5) * (obj._xi + 2 / 5),
                                       (obj._xi - 2 / 5) * (obj._xi - 4 / 5)))
            B *= np.hstack(((-4 / 5 <= obj._xi) & (obj._xi <= -2 / 5),
                            (2 / 5 <= obj._xi) & (obj._xi <= 4 / 5)))
            obj._B = np.abs(B)

            # Forcing term coefficient
            alpha = -obj.kappa * (obj._xi + 1 / 5) * (obj._xi - 1 / 5)
            alpha *= np.abs(obj._xi) <= 1 / 5
            obj._alpha = np.abs(alpha)

        if ('n_states' in new_params or 'kappa' in new_params
                or 'R' in new_params):
            obj.R = float(obj.R)
            obj._RBT = - obj._B.T / (2. * obj.R)

        if not hasattr(obj, '_rng') or 'x0_sample_seed' in new_params:
            obj._rng = np.random.default_rng(
                getattr(obj, 'x0_sample_seed', None))

    def sample_initial_conditions(self, n_samples=1, n_terms=10, distance=None):
        """
        Generate initial conditions by summing a series of sin functions with
        random amplitudes. For each `k` in `range(n_terms)`, the `k`th sin
        function has frequency `(k + 1) * pi` and amplitude uniformly sampled
        between `-1 / (k + 1)` and `1 / (k + 1)`.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        n_terms : int, default=10
            Number of terms in the sine series.
        distance : positive float, optional
            Desired norm of samples, determined by integrating `x0` with
            Clenshaw-Curtis quadrature.

        Returns
        -------
        x0 : (2, n_samples) or (2,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a 1d array.
        """
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive int")

        xi = np.pi * self.parameters._xi
        x0 = np.zeros((self.n_states, n_samples))

        coefs = self.parameters._rng.uniform(low=-1., size=(n_terms, n_samples))

        for k in range(1, n_terms + 1):
            x0 += coefs[k - 1:k] / k * np.sin(k * xi)

        if distance is not None:
            x0 *= distance / self.distances(x0, 0.)

        if n_samples == 1:
            return x0.flatten()
        return x0

    def distances(self, xa, xb, squared=False):
        """
        Calculate the quadrature-integrated distances between a batch of states
        from another state or batch of states.

        Parameters
        ----------
        xa : (n_states, n_a) or (n_states,) array
            First batch of points.
        xb : (n_states, n_b) or (n_states,) array or float
            Second batch of points.
        squared : bool, default=False
            If `squared==True`, the output will be squared.

        Returns
        -------
        dist : (n_a, n_b) array
            For each index `i in range(n_a)` and `j in range(n_b`), `dist[i, j]`
            is the approximate integral between `xa[:, i]` and `xb[:, j]`,
            `sqrt(sum((xa[:, i] - xb[:, j])**2 * self.parameters._w))`. Note
            that the actual computations are not performed with a for loop.
        """
        xa = np.reshape(xa, (self.n_states, -1)).T

        if np.size(xb) > 1:
            xb = np.reshape(xb, (self.n_states, -1)).T
            if squared:
                return cdist(xa, xb, metric='sqeuclidean', w=self.parameters._w)
            return cdist(xa, xb, metric='euclidean', w=self.parameters._w)

        if xb == 0.:
            dist = xa ** 2
        else:
            dist = (xa - xb) ** 2
        dist *= self.parameters._w
        dist = dist.sum(axis=-1, keepdims=True)

        if squared:
            return dist

        return np.sqrt(dist)

    def running_cost(self, x, u):
        x, u, squeeze = self._reshape_inputs(x, u)

        x_err = self.distances(x, 0., squared=True)[:, 0]
        u_err = np.sum(self._saturate(u) ** 2, axis=0)

        L = x_err + self.parameters.R * u_err

        if squeeze:
            return L[0]

        return L

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            if squeeze:
                dLdx = x[:, 0] * (2. * self.parameters._w)
            else:
                dLdx = x * (2. * self.parameters._w.reshape(-1, 1))
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = (2. * self.parameters.R) * self._saturate(u)

            # Where the control is saturated, the gradient is zero
            sat_idx = self._find_saturated(u)
            dLdu[sat_idx] = 0.
            if squeeze:
                dLdu = dLdu[..., 0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            Q = np.diag(self.parameters._w)
            if not squeeze:
                Q = Q[..., None]
            if x.shape[1] > 1:
                Q = np.tile(Q, (1, 1, x.shape[1]))

            if not return_dLdu:
                return Q

        if return_dLdu:
            R = np.diag(np.full(self.n_controls, self.parameters.R))[..., None]
            if u.shape[1] > 1:
                R = np.tile(R, (1, 1, np.shape(u)[1]))

            # Where the control is saturated, the gradient is zero (constant).
            # This makes the Hessian zero in all terms that include a saturated
            # control
            sat_idx = self._find_saturated(u)
            sat_idx = sat_idx[None, ...] + sat_idx[:, None, ...]
            R[sat_idx] = 0.

            if squeeze:
                R = R[..., 0]
            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        x, u, squeeze = self._reshape_inputs(x, u)

        dxdt = (-0.5 * np.matmul(self.parameters._D, x**2)
                + np.matmul(self.parameters.nu * self.parameters._D2, x)
                + x * self.parameters._alpha / np.exp(self.parameters.gamma * x)
                + np.matmul(self.parameters._B, self._saturate(u)))

        if squeeze:
            return dxdt[:, 0]

        return dxdt

    def jac(self, x, u, return_dfdx=True, return_dfdu=True, f0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dfdx:
            dfdx = (x * - self.parameters._D[..., None]
                    + (self.parameters.nu * self.parameters._D2)[..., None])

            # Get writeable view of diagonals of Jacobians
            dfdx_diag = np.einsum('iib->ib', dfdx)

            gamma_x = - self.parameters.gamma * x
            gamma_x = (1. + gamma_x) * self.parameters._alpha * np.exp(gamma_x)

            dfdx_diag += gamma_x

            if squeeze:
                dfdx = dfdx[..., 0]
            if not return_dfdu:
                return dfdx

        if return_dfdu:
            dfdu = np.tile(self.parameters._B[..., None], (1, 1, u.shape[1]))

            # Where the control is saturated, the Jacobian is zero
            sat_idx = self._find_saturated(u)
            dfdu[:, sat_idx] = 0.

            if squeeze:
                dfdu = dfdu[..., 0]
            if not return_dfdx:
                return dfdu

        return dfdx, dfdu

    def optimal_control(self, x, p):
        u = np.matmul(self.parameters._RBT, p)
        return self._saturate(u)

    def optimal_control_jac(self, x, p, u0=None):
        return np.zeros((self.n_controls, *np.shape(x)))

    def bvp_dynamics(self, t, xp):
        # Extract states and costates
        x = xp[:self.n_states]
        p = xp[self.n_states:-1]

        u = self.optimal_control(x, p)
        L = np.atleast_1d(self.running_cost(x, u))

        if np.ndim(x) == 1:
            wx = self.parameters._w * x
            aex = (self.parameters._alpha[:, 0]
                   * np.exp(-self.parameters.gamma * x))
        else:
            wx = self.parameters._w.reshape(-1, 1) * x
            aex = self.parameters._alpha * np.exp(-self.parameters.gamma * x)
            L = L.reshape(1, -1)

        dxdt = (-0.5 * np.matmul(self.parameters._D, x ** 2)
                + np.matmul(self.parameters.nu * self.parameters._D2, x)
                + x * aex
                + np.matmul(self.parameters._B, self._saturate(u)))

        dpdt = (- 2. * wx
                + x * np.matmul(self.parameters._D.T, p)
                - np.matmul(self.parameters.nu * self.parameters._D2.T, p)
                - aex * (1. - self.parameters.gamma * x) * p)

        return np.concatenate((dxdt, dpdt, -L), axis=0)


def cheb(n):
    """
    Build Chebyshev collocation points `x_nodes`, differentiation matrix `D`,
    and Clenshaw-Curtis integration weights `w`. See the algorithms on pages 54
    and 128 of Spectral Methods in MATLAB, Trefethen (2000).

    The Chebyshev collocation points are defined as
    `x_nodes[i] = cos(pi / n * i)` for `i in range(0, n + 1)`. Suppose `y` is an
    `(n + 1,)` array with `y[i]` equal to some function $y = y(x)$ evaluated at
    `x_nodes[i]`. Multiplying `matmul(D, y)` approximates the derivative $dy/dx$
    at `x_nodes`. Conversely, `sum(w * y)` approximates the integral of $y(x)$
    between -1 and 1 (the lower and upper limits of `x_nodes`).
    
    Parameters
    ----------
    n : int
        One less than the number of desired Chebyshev collocation points.

    Returns
    -------
    x_nodes : (n + 1,) array
        Chebyshev collocation points.
    D : (n + 1, n + 1) array
        Chebyshev differentiation matrix.
    w : (n + 1,) array
        Clenshaw-Curtis integration weights.
    """
    theta = np.pi / n * np.arange(n + 1)
    x_nodes = np.cos(theta)

    x = np.tile(x_nodes, (n + 1, 1))
    x = x.T - x

    C = np.ones(n+1)
    C[0] = 2.
    C[-1] = 2.
    C[1::2] = -C[1::2]
    C = np.outer(C, 1. / C)

    D = C / (x + np.identity(n + 1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    w = np.empty_like(x_nodes)
    v = np.ones(n-1)
    for k in range(2, n, 2):
        v -= 2. * np.cos(k * theta[1:-1]) / (k**2 - 1)

    if n % 2 == 0:
        w[0] = 1. / (n**2 - 1)
        v -= np.cos(n * theta[1:-1]) / (n**2 - 1)
    else:
        w[0] = 1. / n**2

    w[-1] = w[0]
    w[1:-1] = 2. * v/n

    return x_nodes, D, w
