import inspect
import mlflow
from typing import Tuple


def mlflow_logging(ignore : Tuple[str], **factory_kwargs):

    def decorator(func):
        def initialize(arg_spec):
            params = {k: None for k in arg_spec.args}

            return params

        def log_default(params, arg_spec):
            for i in range(len(arg_spec.defaults)):
                params[arg_spec.args[-(i + 1)]] = arg_spec.defaults[-(i + 1)]

            return params

        def log_input(params, arg_spec, args, kwargs):
            return {**dict(zip(arg_spec.args, args)), **kwargs, **factory_kwargs, **params}

        def filter_ignore(params):
            for k in ignore:
                params.pop(k, None)

            params.pop('ignore', None)
            params.pop('self', None)
            params.pop('cls', None)

            return params

        def log(*args, **kwargs):

            if not mlflow.active_run():
                print("Not active run")
                return func(*args, **kwargs)

            arg_spec = inspect.getargspec(func)

            params = initialize(arg_spec)
            params = log_default(params, arg_spec)
            params = log_input(params, arg_spec, args, kwargs)
            params = filter_ignore(params)

            mlflow.log_params(params)

            return func(*args, **kwargs)

        return log

    return decorator
