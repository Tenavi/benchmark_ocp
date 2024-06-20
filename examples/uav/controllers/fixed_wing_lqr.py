import numpy as np

from optimalcontrol.controls import LinearQuadraticRegulator

from examples.uav.problem_definition import scale_altitude


class FixedWingLQR(LinearQuadraticRegulator):
    def __init__(self, ocp):
        xf = ocp.parameters.trim_state.to_array()
        uf = ocp.parameters.trim_controls.to_array()

        A, B = ocp.jac(xf, uf)
        Q, R = ocp.running_cost_hess(xf, uf)

        super().__init__(A=A, B=B, Q=Q, R=R, xf=xf, uf=uf,
                         u_lb=ocp.control_lb, u_ub=ocp.control_ub)

        self._h_scale = ocp.parameters.h_cost_ceil

    def __call__(self, x):
        # Rescale altitude so LQR doesn't act badly for large commands
        x_rescale = np.copy(x)
        x_rescale[0] = self._h_scale * scale_altitude(x_rescale[0],
                                                      self._h_scale)
        return super().__call__(x_rescale)

    def jac(self, x, u0=None):
        dudx = super().jac(x, u0=u0)
        dudx[:, 0] /= np.cosh(x[0] / self._h_scale) ** 2
        return dudx
