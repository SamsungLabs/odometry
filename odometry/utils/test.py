from odometry.utils.logging_utils import mlflow_logging


@mlflow_logging
def ff(a, p=1, b=2, **kwargs):
    print(a, p, b)


ff('a', ignore=('p'))
