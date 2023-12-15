import numpy as np


class TimeMapRadau:
    """
    Abstract class for representing mappings between physical time and the
    half-open interval [-1, 1) for use with Radau methods.
    """

    @staticmethod
    def physical_to_radau(t):
        """
        Convert physical time `t` to the half-open interval [-1, 1).

        Parameters
        ----------
        t : (n_points,) array
            Physical time, `t >= 0`.

        Returns
        -------
        tau : (n_points,) array
            Mapped time points in [-1, 1).
        """
        raise NotImplementedError

    @staticmethod
    def radau_to_physical(tau):
        """
        Convert points `tau` from half-open interval [-1, 1) to physical time by
        the inverse map.

        Parameters
        ----------
        tau : (n_points,) array
            Mapped time points in [-1, 1).

        Returns
        -------
        t : (n_points,) array
            Physical time, `t >= 0`.
        """
        raise NotImplementedError

    @staticmethod
    def derivative(tau):
        r"""
        Derivative of the inverse map `radau_to_physical` from Radau points
        `tau` in [-1, 1) to physical time `t`. Used for chain rule.

        Parameters
        ----------
        tau : (n_points,) array
            Mapped time points in [-1, 1).

        Returns
        -------
        dt_dtau (n_points,) array
            Derivative of the inverse time map.
        """
        raise NotImplementedError


class TimeMapRational(TimeMapRadau):
    """
    Maps physical time `t` to the half-open interval [-1, 1) by the map
    `tau = (t - 1) / (t + 1)`. This is the map proposed by Fahroo (2008) (see
    also eq. (2) in Garg (2011)).
    """

    @staticmethod
    def physical_to_radau(t):
        t = np.asarray(t)
        return (t - 1.) / (t + 1.)

    @staticmethod
    def radau_to_physical(tau):
        tau = np.asarray(tau)
        return (1. + tau) / (1. - tau)

    @staticmethod
    def derivative(tau):
        return 2. / (1. - np.asarray(tau)) ** 2


class TimeMapLog2(TimeMapRadau):
    """
    Maps physical time `t` to the half-open interval [-1, 1) by the map
    `tau = 1 - 2 * exp(- t / 2)`. This is the map given by eq. (4) in Garg
    (2011).
    """

    @staticmethod
    def physical_to_radau(t):
        return 1. - 2. * np.exp(-0.5 * np.asarray(t))

    @staticmethod
    def radau_to_physical(tau):
        return np.log(4. / (1. - np.asarray(tau)) ** 2)

    @staticmethod
    def derivative(tau):
        return 2. / (1. - np.asarray(tau))
