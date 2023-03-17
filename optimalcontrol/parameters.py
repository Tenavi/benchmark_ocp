import numpy as np
import warnings

class ProblemParameters:
    """Utility class to store cost function and system dynamics parameters."""
    def __init__(self, required=[], optional=[], update_fun=None):
        """
        Parameters
        ----------
        required : list of strings, default=[]
            List of required parameter names.
        optional : list of strings, default=[]
            List of optional parameter names.
        update_fun : callable, optional
            A function to execute whenever problem parameters are modified by
            `update`. The function must have the call signature
            `update_fun(obj, **params)` where `obj` refers to the
            `ProblemParameters` instance and `params` are parameters to be
            modified, specified as keyword arguments.
        """
        if update_fun is not None:
            self.update_fun = update_fun
        else:
            self._update_fun = None

        self.required = required
        self.optional = optional

    def update(self, check_required=True, **params):
        """
        Modify individual or multiple parameters using keyword arguments.

        Parameters
        ----------
        check_required : bool, default=True
            Raises a `RuntimeError` if `check_required is True` and any of the
            required parameters is `None` after updating.
        **params : dict
            Parameters to change, as keyword arguments. Sets attributes of the
            `ProblemParameters` instance with these parameters, so if these
            overwrite existing class or instance attributes then unexpected
            behavior may occur.
        """
        for key, val in params.items():
            if key in self.required or key in self.optional:
                setattr(self, key, val)
            else:
                warnings.warn(
                    key + " is not in the required or optional parameter lists",
                    category=RuntimeWarning
                )

        if check_required:
            self.check_required()

        # Run other needed operations
        self.update_fun(self, **params)

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
                raise RuntimeError(param + " is required but has not been set")

    @property
    def update_fun(self):
        """
        Get or set a function to execute whenever problem parameters are
        modified by `update`. The function must have the call signature
        `update_fun(obj, **params)` where `obj` refers to the
        `ProblemParameters` instance and `params` are parameters to be modified,
        specified as keyword arguments.
        """
        if not callable(self._update_fun):
            raise RuntimeError("update_fun has not been set")
        return self._update_fun

    @update_fun.setter
    def update_fun(self, update_fun):
        if not callable(update_fun):
            raise TypeError("update_fun must be set with a callable")
        self._update_fun = update_fun

    @property
    def required(self):
        """
        Get or set a list of strings which designate required parameters. Also
        can be set with a dictionary, in which case `self.update` is called.
        """
        return getattr(self, "_required_params", [])

    @property
    def optional(self):
        """
        Get or set a list of strings which designate optional parameters. Also
        can be set with a dictionary, in which case `self.update` is called.
        """
        return getattr(self, "_optional_params", [])

    @required.setter
    def required(self, required_params):
        self._set_param_list(required_params, "_required_params", self.optional)

    @optional.setter
    def optional(self, optional_params):
        self._set_param_list(optional_params, "_optional_params", self.required)

    def _set_param_list(self, param_list, list_attr, other_list):
        """
        Used by the `required` and `optional` `setters` to extend the lists
        of required or optional parameters. Makes sure that the lists each only
        contain one copy of a parameter, and throws an error if the two lists
        conflict.

        Parameters
        ----------
        param_list : list of strings
            New parameter names with which to extend the existing list.
        list_attr : string
            The name of the attribute to set, e.g. `"_required_params"` or
            `"optional_params"`.
        other_list : list of strings
            Another list of strings to compare with for uniqueness.
        """
        param_dict = self._convert_param_dict(param_list)

        for param in param_dict.keys():
            if np.isin(param, other_list):
                raise ValueError(param + " cannot be in both required and optional parameter lists")

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
        elif hasattr(param_dict, "__iter__"):
            param_dict = dict(zip(param_dict, [None]*len(param_dict)))
        else:
            raise TypeError("New parameter list or dict must be iterable")

        for key in param_dict:
            if not isinstance(key, str):
                raise TypeError("Parameter list elements or dict keys must be strings")

        return param_dict
