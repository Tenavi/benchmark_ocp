from itertools import product
from matplotlib import pyplot as plt
import numpy as np

from examples.fixed_wing.vehicle_models.aerosonde import aero


def plot_slices(plot_fun, fun, x, y, xlabel, ylabel, zlabels, **kwargs):
    kwargs = {key: np.reshape(val, -1) for key, val in kwargs.items()}
    n_x = [val.size for val in kwargs.values()]
    if len(n_x) == 0:
        return [plot_fun(fun, zlabels, **{xlabel: x, ylabel: y}, **kwargs)]

    figs = []
    for idx in product(*[range(n_x_i) for n_x_i in n_x]):
        x_fixed = {key: val[idx[i]] for i, (key, val)
                   in enumerate(kwargs.items())}
        figs.append(plot_fun(fun, zlabels, **{xlabel: x, ylabel: y, **x_fixed}))

    return figs


def plot_1d(fun, zlabels, **kwargs):
    if isinstance(zlabels, str):
        zlabels = (zlabels,)
    n_z = len(zlabels)

    x_grid = dict()
    x_fixed = dict()
    for key, val in kwargs.items():
        if np.size(val) == 1:
            x_fixed[key] = np.reshape(val, -1)
        else:
            x_grid[key] = np.reshape(val, -1)
    if len(x_grid) != 2:
        raise ValueError("plot_2d requires exactly 2 kwargs with more than one "
                         "point")

    xlabel, ylabel = x_grid.keys()
    x, y = x_grid.values()

    z = np.empty((n_z, x.size, y.size))

    for j in range(x.size):
        for k in range(y.size):
            z[:, j, k] = fun(**{xlabel: x[j], ylabel: y[k]}, **x_fixed)

    fig, axes = plt.subplots(nrows=n_z, figsize=(6.4, (n_z + 1) * 2),
                             layout='constrained')

    for i, zlabel in enumerate(zlabels):
        for k in range(y.size):
            axes[i].plot(x, z[i, :, k], label=f'{ylabel}={y[k]:.1f}')
            axes[i].set(xlabel=xlabel, ylabel=zlabel)

        axes[i].grid()

        if i == 0:
            axes[i].legend(ncols=y.size // 4)
            axes[i].set_title(', '.join([f'{key}={float(val):.1f}'
                                         for key, val in x_fixed.items()]))

    return fig


def plot_2d(fun, zlabels, **kwargs):
    if isinstance(zlabels, str):
        zlabels = (zlabels,)
    n_z = len(zlabels)

    x_grid = dict()
    x_fixed = dict()
    for key, val in kwargs.items():
        if np.size(val) == 1:
            x_fixed[key] = val
        else:
            x_grid[key] = val
    if len(x_grid) != 2:
        raise ValueError("plot_2d requires exactly 2 kwargs with more than one "
                         "point")

    xlabel, ylabel = x_grid.keys()
    x, y = np.meshgrid(*x_grid.values())

    z = np.empty((n_z,) + x.shape)

    for j in range(x.shape[0]):
        for k in range(x.shape[1]):
            z[:, j, k] = fun(**{xlabel: x[j, k], ylabel: y[j, k]}, **x_fixed)

    title = ', '.join([f'{key}={val:.1f}' for key, val in x_fixed.items()])

    figs = []

    for i, zlabel in enumerate(zlabels):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ax.plot_surface(x, y, z[i], cmap='viridis')
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)

        figs.append(fig)

    return figs


def make_deg_wrapper(fun, *deg_args):
    def wrapped_fun(**kwargs):
        kwargs = {key: np.deg2rad(arg) if key in deg_args else arg
                  for key, arg in kwargs.items()}
        return fun(**kwargs)
    return wrapped_fun


def longitudinal_aero_CL_CD(alpha, va, q, elevator):
    coefs = aero._longitudinal_aero(alpha, va, q, elevator)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    # Rotate axial and normal forces back to lift and drag
    rotation = np.array([[-sin_alpha, cos_alpha],
                         [cos_alpha, sin_alpha]])

    coefs[:2] = np.matmul(rotation, -coefs[:2])

    return coefs


if __name__ == '__main__':
    va = 25.

    p = q = r = np.linspace(-30., 30., 7)
    aileron = elevator = rudder = np.linspace(-45., 45., 7)
    throttle = np.linspace(0., 1., 6)

    alpha = np.linspace(-10., 60., 71)
    zlabels = ['CL', 'CD', 'Cm']
    fun = make_deg_wrapper(longitudinal_aero_CL_CD, 'alpha', 'q', 'elevator')

    plot_slices(plot_1d, fun, alpha, q, 'alpha', 'q', zlabels,
                va=va, elevator=0.)

    plot_slices(plot_1d, fun, alpha, elevator, 'alpha', 'elevator', zlabels,
                va=va, q=0.)

    beta = np.linspace(-10., 10., 21)
    zlabels = ['CY', 'Cl', 'Cn']
    fun = make_deg_wrapper(aero._lateral_aero,
                           'beta', 'p', 'r', 'aileron', 'rudder')

    plot_slices(plot_1d, fun, beta, p, 'beta', 'p', zlabels,
                va=va, r=0., aileron=0., rudder=0.)

    plot_slices(plot_1d, fun, beta, r, 'beta', 'r', zlabels,
                va=va, p=0., aileron=0., rudder=0.)

    plot_slices(plot_1d, fun, beta, aileron, 'beta', 'aileron', zlabels,
                va=va, p=0., r=0., rudder=0.)

    plot_slices(plot_1d, fun, beta, rudder, 'beta', 'rudder', zlabels,
                va=va, p=0., r=0., aileron=0.)

    va = np.linspace(0., 30., 31)

    plot_slices(plot_1d, aero.prop_forces, va, throttle, 'va', 'throttle',
                ['thrust', 'torque'])

    plt.show()
