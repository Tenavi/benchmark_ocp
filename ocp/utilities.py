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

def cross_product_matrix(w):
    zeros = np.zeros_like(w[0])
    wx = np.array([
        [zeros, -w[2], w[1]],
        [w[2], zeros, -w[0]],
        [-w[1], w[0], zeros]]
    )
    return wx
