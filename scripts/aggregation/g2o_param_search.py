import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from .g2o_estimator import G2OEstimator


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def random_search(X, y, param_distributions, **kwargs):
    rs = RandomizedSearchCV(G2OEstimator(verbose=True), 
                            param_distributions,
                            cv=DisabledCV(),
                            refit=False,
                            **kwargs)
    rs.fit(X, y)

    print(f'Best params: {rs.best_params_}')
    print(f'Best score: {rs.best_score_}')

    return pd.DataFrame(rs.cv_results_)
