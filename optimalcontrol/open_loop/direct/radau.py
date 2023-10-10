import numpy as np
from scipy.special import legendre, roots_jacobi


def make_lgr(n_nodes):
    """
    Constructs LGR collocation points, integration weights, and differentiation
    matrix. See `make_lgr_nodes`, `make_lgr_weights`, and `make_lgr_diff_matrix`
    for details.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes. Must be `n_nodes >= 3`.

    Returns
    -------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points `tau`.
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points
        `tau`.
    """
    n_nodes = _check_size_n(n_nodes)

    tau = make_lgr_nodes(n_nodes)

    legendre_eval = legendre(n_nodes - 1)(tau)

    w = make_lgr_weights(tau, legendre_eval=legendre_eval)
    D = make_lgr_diff_matrix(tau, legendre_eval=legendre_eval)

    return tau, w, D


def make_scaled_lgr(n_nodes):
    """
    Constructs LGR collocation points, integration weights, and differentiation
    matrix. The weights are scaled by `inverse_time_map_deriv(tau)`, the
    derivative of the mapping from physical time to LGR points, and the
    differentiation matrix is scaled by `1 / inverse_time_map_deriv(tau)`.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes. Must be `n_nodes >= 3`.

    Returns
    -------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points `tau`,
        scaled by `inverse_time_map_deriv(tau)`.
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points
        `tau`, scaled by `1 / inverse_time_map_deriv(tau)`.
    """
    tau, w, D = make_lgr(n_nodes)

    r_tau = inverse_time_map_deriv(tau)
    w = w * r_tau
    D = np.einsum('i,ij->ij', 1. / r_tau, D)

    return tau, w, D


def make_lgr_nodes(n):
    r"""
    Constructs collocation points for LGR quadrature. These are the roots of
    $P_n(\tau) + P_{n-1}(\tau)$, where $P_n$ is the `n`th order Legendre
    polynomial. One can show (see e.g.
    https://mathworld.wolfram.com/JacobiPolynomial.html) that
    $P_n(\tau) + P_{n-1}(\tau) = (1 + \tau) P^{(0,1)}_n (\tau)$, where
    $P^{(0,1)}_n$ is the `n`th order Jacobi polynomial with
    $\alpha = 0, \beta = 1$.

    Parameters
    ----------
    n : int
        Number of collocation nodes. Must be `n >= 3`.

    Returns
    -------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    """
    n = _check_size_n(n)
    tau, _ = roots_jacobi(n - 1, alpha=0, beta=1)
    return np.concatenate((np.array([-1.]), tau))


def make_lgr_weights(tau, legendre_eval=None):
    """
    Constructs the LGR quadrature weights, `w`. The entries of `w` are given by
    ```
    w[0] = 2 / n ** 2
    ```
    and
    ```
    w[i] = (1 - tau[i]) / (n ** 2 + legendre(n - 1)(tau[i])),
    ```
    for `i = 1, ..., n - 1` where `n = n_nodes = tau.shape[0]` is the number of
    collocation points and `legendre(n - 1)` is the (`n - 1`)th order Legendre
    polynomial, $P_{n-1}$.

    Parameters
    ----------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    legendre_eval : (n_nodes,) array, optional
        Pre-computed values for `scipy.special.legendre(n_nodes - 1)(tau)`.

    Returns
    -------
    w : (n_nodes,) array
        LGR quadrature weights corresponding to the collocation points `tau`.
    """
    n = _check_size_n(tau.shape[0])

    if legendre_eval is None:
        legendre_eval = legendre(n - 1)(tau)

    w = np.empty_like(tau)
    w[0] = 2. / n ** 2
    for i in range(1, n):
        w[i] = (1. - tau[i]) / (n * legendre_eval[i])**2
    return w


def make_lgr_diff_matrix(tau, legendre_eval=None):
    """
    Constructs the LGR differentiation_matrix, `D`. The entries of `D` are given
    by
    ```
    D[i, j] = - (n - 1) * (n + 1) /4
    ```
    for `i == j == 0`,
    ```
    D[i, j] = 1 / (2 - 2 * tau[i])
    ```
    for `1 <= i == j <= n - 1`, and
    ```
    numerator = legendre(n - 1)(tau[i]) * (1 - tau[j])
    denominator = legendre(n - 1)(tau[j]) * (1 - tau[i]) * (tau[i] - tau[j])
    D[i, j] = numerator / denominator
    ```
    otherwise, where `n = n_nodes = tau.shape[0]` is the number of collocation
    points and `legendre(n - 1)` is the (`n - 1`)th order Legendre polynomial,
    $P_{n-1}$.

    Parameters
    ----------
    tau : (n_nodes,) array
        LGR collocation nodes on [-1, 1).
    legendre_eval : (n_nodes,) array, optional
        Pre-computed values of `scipy.special.legendre(n_nodes - 1)(tau)`.

    Returns
    -------
    D : (n_nodes, n_nodes) array
        LGR differentiation matrix corresponding to the collocation points
        `tau`.
    """
    n = _check_size_n(tau.shape[0])

    if legendre_eval is None:
        legendre_eval = legendre(n - 1)(tau)

    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                num = legendre_eval[i] * (1. - tau[j])
                den = legendre_eval[j] * (1. - tau[i]) * (tau[i] - tau[j])
                D[i, j] = num / den
            elif i == j == 0:
                D[i, j] = - (n - 1) * (n + 1) / 4.
            else:
                D[i, j] = 1. / (2. * (1. - tau[i]))
    return D


def time_map(t):
    """
    Convert physical time `t` to the half-open interval [-1, 1) by the map
    ```
    tau = (t - 1) / (t + 1)
    ```

    Parameters
    ----------
    t : (n_points,) array
        Physical time, `t >= 0`.

    Returns
    -------
    tau : (n_points,) array
        Mapped time points in [-1, 1).
    """
    t = np.asarray(t)
    return (t - 1.) / (t + 1.)


def inverse_time_map(tau):
    """
    Convert points `tau` from half-open interval [-1, 1) to physical time by the
    map
    ```
    t = (1 + tau) / (1 - tau)
    ```

    Parameters
    ----------
    tau : (n_points,) array
        Mapped time points in [-1, 1).

    Returns
    -------
    t : (n_points,) array
        Physical time, `t >= 0`.
    """
    tau = np.asarray(tau)
    return (1. + tau) / (1. - tau)


def inverse_time_map_deriv(tau):
    r"""
    Derivative of the `inverse_time_map` from Radau points `tau` in [-1, 1) to
    physical time `t`. Used for chain rule.

    Parameters
    ----------
    tau : (n_points,) array
        Mapped time points in [-1, 1).

    Returns
    -------
    r : (n_points,) array
        Derivative of the inverse time map,
        $r(\tau) = dt/d\tau = 2 / (1 - \tau)^2$.
    """
    return 2. / (1. - np.asarray(tau)) ** 2


def _check_size_n(n_nodes):
    """
    We only define the LGR quadrature for `n_nodes >= 3`. This utility function
    checks to make sure `n_nodes` is the right size.

    Parameters
    ----------
    n_nodes : int
        Number of collocation nodes.

    Returns
    -------
    n_nodes : int
        Number of collocation nodes, only returned if `n_nodes >= 3`.

    Raises
    ------
    ValueError
        If `n_nodes < 3`.
    """
    n_nodes = int(n_nodes)
    if n_nodes < 3:
        raise ValueError("Number of nodes must be at least n_nodes >= 3.")
    return n_nodes
