import numpy as np
import warnings


class ProblemParameters:
    """Utility class to store cost function and system dynamics parameters and
    allow these to be updated (during simulation, for example)."""
    def __init__(self, required=[], optional=[], update_fun=None):
        """
        Parameters
        ----------
        required : list of strings, default=[]
            Names of parameters required by the optimal control problem.
        optional : list of strings, default=[]
            Names of optional problem parameters.
        update_fun : callable, optional
            A function to execute whenever problem parameters are modified by
            `update`. The function must have the call signature
            `update_fun(obj, **params)` where `obj` refers to the
            `ProblemParameters` instance and `params` are parameters to be
            modified, specified as keyword arguments.
        """
        if update_fun is None:
            self._update_fun = lambda s, **params: None
        elif not callable(update_fun):
            raise TypeError('update_fun must be set with a callable')
        else:
            self._update_fun = update_fun

        self.required = required
        self.optional = optional

    def update(self, check_required=True, **params):
        """
        Modify individual or multiple parameters using keyword arguments. This
        internally calls `self._update_fun(self, **params)`.

        Parameters
        ----------
        check_required : bool, default=True
            Raises a `RuntimeError` if `check_required is True` and any of the
            required parameters is `None` after updating.
        **params : dict
            Parameters to change, as keyword arguments. Sets attributes of the
            `ProblemParameters` instance with these parameters, so unexpected
            behavior can occur if these overwrite class or instance attributes.
        """
        for key, val in params.items():
            if key in self.required or key in self.optional:
                setattr(self, key, val)
            else:
                warnings.warn(f'{key} is not in the required or optional '
                              f'parameter lists', category=RuntimeWarning)

        if check_required:
            self.check_required()

        # Run other needed operations
        self._update_fun(self, **params)

    def check_required(self):
        """
        Check that all required parameters have been set.

        Raises
        ------
        RuntimeError
            If any of the parameters listed in `self.required` is `None`.
        """
        for param in self.required:
            if getattr(self, param, None) is None:
                raise RuntimeError(f'{param} is required but has not been set')

    @property
    def required(self):
        """
        Get or set a list of strings which designate required parameters. Can be
        set with a dict, in which case `self.update` is called.
        """
        return getattr(self, '_required_params', [])

    @property
    def optional(self):
        """
        Get or set a list of strings which designate optional parameters. Can be
        set with a dict, in which case `self.update` is called.
        """
        return getattr(self, '_optional_params', [])

    @required.setter
    def required(self, required_params):
        self._set_param_list(required_params, '_required_params', self.optional)

    @optional.setter
    def optional(self, optional_params):
        self._set_param_list(optional_params, '_optional_params', self.required)

    def _set_param_list(self, param_list, list_attr, other_list):
        """
        Used by the `required` and `optional` setters to extend the lists of
        parameters. Makes sure that the lists each contain only one copy of each
        parameter, and throws an error if the two lists conflict.

        Parameters
        ----------
        param_list : list of strings
            New parameter names with which to extend the existing list.
        list_attr : string
            The name of the attribute to set, e.g. `'_required_params'` or
            `'_optional_params'`.
        other_list : list of strings
            Another list of strings to compare with for uniqueness.
        """
        param_dict = self._convert_param_dict(param_list)

        for param in param_dict.keys():
            if np.isin(param, other_list):
                raise ValueError(f'{param} cannot be in both required and '
                                 f'optional parameter lists')

        if hasattr(self, list_attr):
            param_list = getattr(self, list_attr) + list(param_dict.keys())
            param_list = np.unique(param_list).tolist()
            setattr(self, list_attr, param_list)
        else:
            setattr(self, list_attr, list(param_dict.keys()))

        if np.any([val is not None for val in param_dict.values()]):
            self.update(check_required=False, **param_dict)

    @staticmethod
    def _convert_param_dict(param_dict):
        """Convert a dict or other iterable of strings into a dict, where
        values are `None` if the original input is not already a dict."""
        if isinstance(param_dict, dict):
            pass
        elif isinstance(param_dict, str):
            param_dict = {param_dict: None}
        elif hasattr(param_dict, '__iter__'):
            param_dict = dict(zip(param_dict, [None]*len(param_dict)))
        else:
            raise TypeError('New parameter list or dict must be iterable')

        for key in param_dict:
            if not isinstance(key, str):
                raise TypeError('Parameter list elements or dict keys must be'
                                'strings')

        return param_dict
