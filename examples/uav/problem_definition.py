import numpy as np

from optimalcontrol.problem import OptimalControlProblem
from optimalcontrol.sampling import UniformSampler

from examples.common_utilities.dynamics import (euler_to_quaternion,
                                                quaternion_to_euler)

from .fixed_wing_dynamics.containers import VehicleState, Controls
from .fixed_wing_dynamics.trim import compute_trim
from .fixed_wing_dynamics.dynamics import dynamics as dynamics_fun
from examples.uav.vehicle_models import aerosonde


_va_target_default = 25.
_h_cost_ceil_default = 50.
_Q_default = VehicleState(pd=_h_cost_ceil_default ** -2,
                          u=_va_target_default ** -2,
                          v=1.,
                          w=1.,
                          p=np.deg2rad(30.) ** -2,
                          q=np.deg2rad(30.) ** -2,
                          r=np.deg2rad(30.) ** -2,
                          attitude=[1., 1., 1., 0.]).to_array()
_R_default = (aerosonde.constants.max_controls.to_array()
              - aerosonde.constants.min_controls.to_array()) ** -2
_x0_max_perturb_default = VehicleState(pd=100.,
                                       u=5.,
                                       v=5.,
                                       w=5.,
                                       p=np.deg2rad(30.),
                                       q=np.deg2rad(30.),
                                       r=np.deg2rad(30.),
                                       attitude=euler_to_quaternion(
                                           [180., 15., 30.], degrees=True))


class FixedWing(OptimalControlProblem):
    _required_parameters = {'vehicle_parameters': aerosonde.constants,
                            'aero_model': aerosonde.aero_model,
                            'va_target': _va_target_default,
                            'h_cost_ceil': _h_cost_ceil_default,
                            'Q': _Q_default,
                            'R': _R_default,
                            'x0_max_perturb': _x0_max_perturb_default}
    _optional_parameters = {'x0_sample_seed': None}

    @property
    def n_states(self):
        return VehicleState.dim

    @property
    def n_controls(self):
        return Controls.dim

    @property
    def final_time(self):
        return np.inf

    @property
    def state_lb(self):
        """(`n_states`,) array. Lower bounds on `pd` (upper bound on altitude)
        and quaternion states, specifying that the scalar quaternion must be
        positive."""
        x_lb = VehicleState(pd=-2.*np.abs(self.parameters.x0_max_perturb.pd),
                            u=-np.inf, v=-np.inf, w=-np.inf,
                            p=-np.inf, q=-np.inf, r=-np.inf,
                            attitude=[-1., -1., -1., 0.])
        return x_lb.to_array()

    @property
    def state_ub(self):
        """(`n_states`,) array. Upper bound on `pd`, translating to a lower
        bound on altitude."""
        x_ub = VehicleState(pd=2. * np.abs(self.parameters.x0_max_perturb.pd),
                            u=np.inf, v=np.inf, w=np.inf,
                            p=np.inf, q=np.inf, r=np.inf,
                            attitude=[1., 1., 1., 1.])
        return x_ub.to_array()

    @staticmethod
    def _parameter_update_fun(obj, **new_params):
        if 'vehicle_parameters' in new_params:
            obj.control_lb = obj.vehicle_parameters.min_controls.to_array(
                copy=True)
            obj.control_ub = obj.vehicle_parameters.max_controls.to_array(
                copy=True)

        if 'va_target' in new_params:
            obj.trim_state, obj.trim_controls, dxdt = compute_trim(
                obj.va_target, obj.vehicle_parameters, obj.aero_model)

            # Confirm that aircraft is in trim
            if not np.allclose(dxdt.to_array(), 0., atol=1e-02):
                raise RuntimeError(f"Trim not achieved for "
                                   f"va_target={obj.va_target:.1f}")

        for var in ('Q', 'R'):
            if var in new_params:
                var_val = getattr(obj, var)
                setattr(obj, var + '_2_diag', np.diag(var_val / 2.))
                setattr(obj, var, var_val.reshape(-1, 1))

        if any([not hasattr(obj, '_x0_sampler'),
                'x0_sample_seed' in new_params,
                'x0_max_perturb' in new_params]):
            xf = np.concatenate([obj.trim_state.to_array()[:-4],
                                 quaternion_to_euler(obj.trim_state.attitude)])
            x0_perturb = np.concatenate([
                obj.x0_max_perturb.to_array()[:-4],
                quaternion_to_euler(obj.x0_max_perturb.attitude)])

            if not hasattr(obj, '_x0_sampler'):
                obj._x0_sampler = UniformSampler(
                    lb=xf - x0_perturb, ub=xf + x0_perturb, xf=xf, norm=np.inf,
                    seed=getattr(obj, 'x0_sample_seed', None))
            else:
                obj._x0_sampler.update(
                    lb=xf - x0_perturb, ub=xf + x0_perturb, xf=xf,
                    seed=new_params.get('attitude_sample_seed', None))

    def sample_initial_conditions(self, n_samples=1, distance=None):
        """
        Generate initial conditions. Euler angles yaw, pitch, roll are sampled
        uniformly from a hypercube, then converted to quaternions, while other
        states are sampled uniformly from a hypercube. Optionally, initial
        conditions may be sampled with a specified distance from equilibrium.

        Parameters
        ----------
        n_samples : int, default=1
            Number of sample points to generate.
        distance : positive float, optional
            Desired infinity norm of initial condition. Note that depending on
            how `distance` is specified, samples may be outside the hypercube
            defined by `self.parameters.x0_max_perturb`.

        Returns
        -------
        x0 : (11, n_samples) or (11,) array
            Samples of the system state, where each column is a different
            sample. If `n_samples==1` then `x0` will be a 1d array.
        """

        # Sample from the hypercube
        x0 = self.parameters._x0_sampler(n_samples=n_samples,
                                         distance=distance)

        # Convert Euler angles to quaternions
        angles = x0[-3:]
        quat = euler_to_quaternion(angles)
        # Set scalar quaternion positive
        quat[-1] = np.abs(quat[-1])

        x0 = np.concatenate([x0[:-3], quat], axis=0)

        return x0

    def running_cost(self, x, u):
        x_err, u_err, squeeze = self._center_inputs(
            x, u, self.parameters.trim_state.to_array(),
            self.parameters.trim_controls.to_array())

        x_err[0] = self.parameters.h_cost_ceil * np.tanh(
            x_err[0] / self.parameters.h_cost_ceil)

        L = (np.sum((self.parameters.Q / 2.) * x_err ** 2, axis=0)
             + np.sum((self.parameters.R / 2.) * u_err ** 2, axis=0))

        if squeeze:
            return L[0]

        return L

    def running_cost_grad(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x_err, u_err, squeeze = self._center_inputs(
            x, u, self.parameters.trim_state.to_array(),
            self.parameters.trim_controls.to_array())

        if return_dLdx:
            # Chain rule for tanh
            x_err[0] /= self.parameters.h_cost_ceil
            x_err[0] = (self.parameters.h_cost_ceil
                        * np.sinh(x_err[0])
                        / np.cosh(x_err[0]) ** 3)

            dLdx = self.parameters.Q * x_err

            if squeeze:
                dLdx = dLdx[..., 0]
            if not return_dLdu:
                return dLdx

        if return_dLdu:
            dLdu = self.parameters.R * u_err

            if squeeze:
                dLdu = dLdu[..., 0]
            if not return_dLdx:
                return dLdu

        return dLdx, dLdu

    def running_cost_hess(self, x, u, return_dLdx=True, return_dLdu=True,
                          L0=None):
        x, u, squeeze = self._reshape_inputs(x, u)

        if return_dLdx:
            # Chain rule for tanh
            h_err = x[0] / self.parameters.h_cost_ceil
            h_err = (1. - 2. * np.sinh(h_err) ** 2) / np.cosh(h_err) ** 4

            if squeeze:
                Q = self.parameters.Q_2_diag.copy()

                Q[0, 0] *= h_err[0]
            else:
                Q = self.parameters.Q_2_diag[..., None]

                if x.shape[1] > 1:
                    Q = np.tile(Q, (1, 1, np.shape(x)[1]))

                Q[0, 0] *= h_err

            if not return_dLdu:
                return Q

        if return_dLdu:
            R = self.parameters.R_2_diag

            if not squeeze:
                R = R[..., None]

                if u.shape[1] > 1:
                    R = np.tile(R, (1, 1, np.shape(u)[1]))

            if not return_dLdx:
                return R

        return Q, R

    def dynamics(self, x, u):
        dxdt = dynamics_fun(VehicleState(array=x), Controls(array=u),
                            self.parameters.vehicle_parameters,
                            self.parameters.aero_model)
        return dxdt.to_array().reshape(x.shape)
