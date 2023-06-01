class ProblemParameters:
    """Utility class to store cost function and system dynamics parameters and
    allow these to be updated (during simulation, for example)."""
    def __init__(self, required=[], update_fun=None, **params):
        """
        Parameters
        ----------
        required : list or set of strings, default=[]
            Names of parameters which cannot be None.
        update_fun : callable, optional
            A function to execute whenever problem parameters are modified by
            `update`. The function must have the call signature
            `update_fun(obj, **params)` where `obj` refers to the
            `ProblemParameters` instance and `params` are parameters to be
            modified, specified as keyword arguments.
        **params : dict
            Parameters to set as initialization, as keyword arguments. Sets
            attributes of the `ProblemParameters` instance with these
            parameters, so unexpected behavior can occur if these overwrite
            class or instance attributes.
        """
        if update_fun is None:
            self._update_fun = lambda s, **p: None
        elif callable(update_fun):
            self._update_fun = update_fun
        else:
            raise TypeError('update_fun must be set with a callable')

        self._param_dict = dict()
        self.required = set(required)
        if len(params):
            self.update(**params)

    def update(self, check_required=True, **params):
        """
        Modify individual or multiple parameters using keyword arguments. This
        internally calls `self._update_fun(self, **params)`.

        Parameters
        ----------
        check_required : bool, default=True
            Ensure that all required parameters have been set (after updating).
        **params : dict
            Parameters to change, as keyword arguments. Sets attributes of the
            `ProblemParameters` instance with these parameters, so unexpected
            behavior can occur if these overwrite class or instance attributes.

        Raises
        ------
        RuntimeError
            If `check_required` is True and any of the parameters in
            `self.required` is None after updating.
        """
        self._param_dict.update(params)
        self.__dict__.update(params)

        if check_required:
            for p in self.required:
                if getattr(self, p, None) is None:
                    raise RuntimeError(f"{p} is required but has not been set")

        # Run other needed operations
        self._update_fun(self, **params)

    def as_dict(self):
        """
        Return all named parameters in the form of a dict.

        Returns
        -------
        parameter_dict : dict
            Dict containing all parameters set using `__init__` or `update`.
        """
        return self._param_dict
