import numpy as np
from scipy.spatial.transform import Rotation


class Container:
    """Base class for state and control containers."""
    dim = None

    def __init__(self, array):
        lengths = np.unique([np.size(arg) for arg in array])
        if np.size(lengths) > 1:
            if lengths[0] > 1 or np.size(lengths) > 2:
                raise ValueError(f"Tried to set components of {type(self)} with"
                                 f" arrays with shapes that cannot be "
                                 f"broadcast to one another")
            for i, arg in enumerate(array):
                if arg.size == 1:
                    array[i] = np.resize(arg, lengths[-1])

        self._array = np.asarray(array).reshape(self.dim, -1)

        if self._array.size == 0:
            raise ValueError(f"{type(self)} cannot be set with empty arrays")

    @property
    def n_points(self):
        return self._array.shape[-1]

    @classmethod
    def from_array(cls, array):
        """
        Initialize the container from a numpy array.

        Parameters
        ----------
        array : (dim,) or (dim, n_points) array
            The vehicle state or control(s).

        Returns
        -------
        container : Container
            The vehicle state or control container initialized with `array`.
        """
        return cls(array=array)

    def to_array(self, copy=False):
        """
        Represent the container as a numpy array.

        Parameters
        ----------
        copy : bool, default=False
            If copy=False (default), returns a reference to the container array.
            Modifying that reference in place will also modify the vehicle
            states or controls. If copy=True, returns a copy of the array
            instead.

        Returns
        -------
        array : (dim,) or (dim, n_points) array
            Array containing the vehicle state or control(s).
        """
        if copy:
            return np.squeeze(self._array).copy()
        return np.squeeze(self._array)

    def __neg__(self):
        return type(self).from_array(-self.to_array())

    def __eq__(self, other):
        return np.array_equal(self.to_array(), other.to_array())

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self + other.to_array()
        else:
            return type(self).from_array(self.to_array() + other)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self - other.to_array()
        else:
            return type(self).from_array(self.to_array() - other)

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self * other.to_array()
        else:
            return type(self).from_array(self.to_array() * other)

    def __truediv__(self, other):
        if isinstance(other, type(self)):
            return self / other.to_array()
        else:
            return type(self).from_array(self.to_array() / other)

    def __pow__(self, exp):
        if isinstance(exp, type(self)):
            return self ** exp.to_array()
        else:
            return type(self).from_array(self.to_array() ** exp)

    def __radd__(self, other):
        if isinstance(other, type(self)):
            return other.to_array() + self
        else:
            return type(self).from_array(other + self.to_array())

    def __rsub__(self, other):
        if isinstance(other, type(self)):
            return other.to_array() - self
        else:
            return type(self).from_array(other - self.to_array())

    def __rmul__(self, other):
        if isinstance(other, type(self)):
            return other.to_array() * self
        else:
            return type(self).from_array(other * self.to_array())

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self._array += other.to_array().reshape(self.dim, -1)
        else:
            self._array += other
        return self

    def __imul__(self, other):
        if isinstance(other, type(self)):
            self._array *= other.to_array().reshape(self.dim, -1)
        else:
            self._array *= other
        return self

    def __idiv__(self, other):
        if isinstance(other, type(self)):
            self._array /= other.to_array().reshape(self.dim, -1)
        else:
            self._array /= other
        return self


class VehicleState(Container):
    """Container holding the vehicle state(s).

    When initializing or representing the `VehicleState` with an array,
    the following order is expected:
    ```
    array[0] = pd
    array[1] = u
    array[2] = v
    array[3] = w
    array[4] = p
    array[5] = q
    array[6] = r
    array[7:11] = attitude
    ```
    """
    dim = 11

    def __init__(self, pd=0., u=0., v=0., w=0., p=0., q=0., r=0.,
                 attitude=[0., 0., 0., 1.], array=None):
        if array is None:
            array = [pd, u, v, w, p, q, r] + list(attitude)
            array = [np.reshape(arg, -1) for arg in array]

        super().__init__(array)

        self._airspeed = None
        self._course = None
        self._rot_mat = None

    pd = property(lambda self: _generic_array_getter(self, 0),
                  lambda self, val: _generic_array_setter(self, val, 0))
    pd.__doc__ = ("(n_points,) array. Inertial down position (negative "
                  "altitude) [m].")

    u = property(lambda self: _generic_array_getter(self, 1),
                 lambda self, val: _generic_array_setter(
                     self, val, 1, '_airspeed', '_course'))
    u.__doc__ = "(n_points,) array. Velocity in body x-axis [m/s]."

    v = property(lambda self: _generic_array_getter(self, 2),
                 lambda self, val: _generic_array_setter(
                     self, val, 2, '_airspeed', '_course'))
    v.__doc__ = "(n_points,) array. Velocity in body y-axis [m/s]."

    w = property(lambda self: _generic_array_getter(self, 3),
                 lambda self, val: _generic_array_setter(
                     self, val, 3, '_airspeed', '_course'))
    w.__doc__ = "(n_points,) array. Velocity in body z-axis [m/s]."

    p = property(lambda self: _generic_array_getter(self, 4),
                 lambda self, val: _generic_array_setter(self, val, 4))
    p.__doc__ = "(n_points,) array. Angular rate about body-x axis [rad/s]."

    q = property(lambda self: _generic_array_getter(self, 5),
                 lambda self, val: _generic_array_setter(self, val, 5))
    q.__doc__ = "(n_points,) array. Angular rate about body-y axis [rad/s]."

    r = property(lambda self: _generic_array_getter(self, 6),
                 lambda self, val: _generic_array_setter(self, val, 6))
    r.__doc__ = "(n_points,) array. Angular rate about body-z axis [rad/s]."

    attitude = property(lambda self: np.squeeze(
                            _generic_array_getter(self, range(7, 11))),
                        lambda self, val: _generic_array_setter(
                            self, np.reshape(val, (4, -1)), range(7, 11),
                            '_course', '_rot_mat'))
    attitude.__doc__ = ("(4,) or (4, n_points,) array. Quaternion of vehicle "
                        "attitude relative to the inertial frame. The vector "
                        "components are indices `attitude[:3]` and the scalar "
                        "quaternion is in `attitude[3]`.")

    @property
    def velocity(self):
        """Reference to subset of `to_array()` containing the vehicle's velocity
        in the body frame, `u`, `v`, and `w`."""
        return self.to_array()[1:4]

    @property
    def rates(self):
        """Reference to subset of `to_array()` containing the vehicle's body
        rotation rates, `p`, `q`, and `r`."""
        return self.to_array()[4:7]

    @property
    def airspeed(self):
        """
        Get current airspeed magnitude, angle of attack, and sideslip, assuming
        zero wind.

        Returns
        -------
        Va : (n_points,) array
            Airspeed [m/s] for each state.
        alpha : (n_points,) array
            Angle of attack [rad] for each state.
        beta : (n_points,) array
            Sideslip [rad] for each state.
        """
        if self._airspeed is None:
            self._airspeed = np.zeros((3, self.n_points))

            # Va
            self._airspeed[0] = np.sqrt(np.einsum('i...,i...->...',
                                                  self.velocity, self.velocity))
            # alpha
            self._airspeed[1] = np.arctan2(self.w, self.u)
            # beta (avoid dividing by zero)
            idx = self._airspeed[0] > 1e-14
            self._airspeed[2, idx] = np.arcsin(
                self.v[idx] / self._airspeed[0, idx])

        return self._airspeed

    @property
    def course(self):
        """
        Get the current course angle, computed by rotating the body velocity
        into the inertial frame.

        Returns
        -------
        chi : (n_points,) array
            Course angle [rad] for each state.
        """
        if self._course is None:
            vel_i = self.body_to_inertial(self.velocity)
            self._course = np.arctan2(vel_i[1], vel_i[0])

        return self._course

    @property
    def rotation_matrix(self):
        """Get an array containing the rotation matrix or matrices representing
        the vehicle's attitude."""
        if self._rot_mat is None:
            rotation = Rotation(self.attitude.reshape(4, -1).T, copy=False)
            self._rot_mat = np.moveaxis(rotation.as_matrix(), 0, -1)
        return self._rot_mat

    def inertial_to_body(self, vec_inertial):
        """
        Rotate a vector from the inertial frame to the vehicle's body frame.

        Parameters
        ----------
        vec_inertial : (3, n_points) or (3,) array
            Vector(s) expressed in the inertial frame.

        Returns
        -------
        vec_body : (3, n_points) array
            `vec_inertial` expressed in the body frame.
        """
        vec_inertial = np.asarray(vec_inertial)
        shape = vec_inertial.shape
        vec_inertial = vec_inertial.reshape(3, -1)

        vec_body = np.einsum('ijb,ib->jb', self.rotation_matrix, vec_inertial)

        # Make the output shape the same as the input, if it didn't increase in
        # size due to multiple rotations.
        if vec_body.shape[1] > vec_inertial.shape[1]:
            return vec_body
        return vec_body.reshape(shape)

    def body_to_inertial(self, vec_body):
        """
        Rotate a vector from the vehicle's body frame to the inertial frame.

        Parameters
        ----------
        vec_body : (3, n_points) or (3,) array
            Vector(s) expressed in the body frame.

        Returns
        -------
        vec_inertial : (3, n_points) array
             `vec_body` expressed in the inertial frame.
        """
        vec_body = np.asarray(vec_body)
        shape = vec_body.shape
        vec_body = vec_body.reshape(3, -1)

        vec_inertial = np.einsum('ijb,jb->ib', self.rotation_matrix, vec_body)

        # Make the output shape the same as the input, if it didn't increase in
        # size due to multiple rotations.
        if vec_inertial.shape[1] > vec_body.shape[1]:
            return vec_inertial
        return vec_inertial.reshape(shape)


class Controls(Container):
    """Container holding the vehicle controls(s).

    When initializing or representing the `Controls` container with an array,
    the following order is expected:
    ```
    array[0] = throttle
    array[1] = aileron
    array[2] = elevator
    array[3] = rudder
    ```
    """
    dim = 4

    def __init__(self, throttle=0., aileron=0., elevator=0., rudder=0.,
                 array=None):
        if array is None:
            array = [throttle, aileron, elevator, rudder]

        super().__init__(array)

    throttle = property(lambda self: _generic_array_getter(self, 0),
                        lambda self, val: _generic_array_setter(self, val, 0))
    throttle.__doc__ = ("(n_points,) array. Throttle setting (increases motor "
                        "speed).")

    aileron = property(lambda self: _generic_array_getter(self, 1),
                       lambda self, val: _generic_array_setter(self, val, 1))
    aileron.__doc__ = "(n_points,) array. Aileron position [rad]."

    elevator = property(lambda self: _generic_array_getter(self, 2),
                        lambda self, val: _generic_array_setter(self, val, 2))
    elevator.__doc__ = "(n_points,) array. Elevator position [rad]."

    rudder = property(lambda self: _generic_array_getter(self, 3),
                      lambda self, val: _generic_array_setter(self, val, 3))
    rudder.__doc__ = "(n_points,) array. Rudder position [rad]."

    def saturate(self, lb, ub, inplace=False):
        """
        Saturate the controls container between lower and upper bound control
        containers.

        Parameters
        ----------
        lb : `Controls`
            Container of lower control bounds. Must satisfy `lb.n_points == 1`
            or `lb.n_points == self.n_points`.
        ub : `Controls`
            Container of upper control bounds. Must satisfy `ub.n_points == 1`
            or `ub.n_points == self.n_points`.
        """

        if inplace:
            self.__init__(array=np.clip(self._array, lb._array, ub._array))
            return self

        return Controls.from_array(np.clip(self._array, lb._array, ub._array))


def _generic_array_getter(obj, idx):
    return obj._array[idx]


def _generic_array_setter(obj, val, idx, *reset_attrs):
    obj._array[idx] = val
    for attr_name in reset_attrs:
        setattr(obj, attr_name, None)
