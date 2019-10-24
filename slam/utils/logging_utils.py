import inspect
import mlflow
from typing import Tuple
from functools import wraps


def mlflow_logging(ignore: Tuple[str] = (), prefix: str = '', **kwargs):

    def decorator(func):
        def log_default(arg_spec):
            if arg_spec.defaults is None:
                return dict()
            else:
                return dict(zip(arg_spec.args[::-1], arg_spec.defaults[::-1]))

        def log_input(default_params, arg_spec, input_args, input_kwargs):
            assert not set(input_kwargs.keys()) & set(kwargs.keys())
            input_params = dict(zip(arg_spec.args, input_args))
            return {**default_params, **input_params, **input_kwargs, **kwargs}

        def filter_ignore(params):
            for k in ignore:
                params.pop(k, None)

            params.pop('ignore', None)
            params.pop('self', None)
            params.pop('cls', None)

            return params

        def add_prefix(params):
            return {prefix + k: v for k, v in params.items()}

        @wraps(func)
        def wrapper(*args, **wrapper_kwargs):

            if not mlflow.active_run():
                return func(*args, **wrapper_kwargs)

            arg_spec = inspect.getargspec(func)

            params = log_default(arg_spec)
            params = log_input(params, arg_spec, args, wrapper_kwargs)
            params = filter_ignore(params)
            params = add_prefix(params)

            mlflow.log_params(params)

            return func(*args, **wrapper_kwargs)

        return wrapper

    return decorator
