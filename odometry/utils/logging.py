import inspect
import mlflow


def mlflow_logging(func):

    def initialize(arg_spec):
        params = {k: None for k in arg_spec.args}

        return params

    def log_default(params, arg_spec):
        for i in range(len(arg_spec.defaults)):
            params[arg_spec.args[-(i + 1)]] = arg_spec.defaults[-(i + 1)]

        return params

    def log_input(params, arg_spec, args, kwargs):
        for i in range(len(args)):
            params[arg_spec.args[i]] = args[i]

        for k, v in kwargs.items():
            params[k] = v

        return params

    def filter_ignore(params):
        ignore = params.get('ignore', list())
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

        print(params)

        mlflow.log_params(params)

        return func(*args, **kwargs)

    return log
