import numbers
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from functools import partial, update_wrapper

from .g2o_estimator import G2OEstimator


def _multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring.
       Original function calculates predict for every score. This function evaluate predict only one time"""

    averaged_metrics = estimator.predict(X_test, y_test)

    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            raise RuntimeError('Not supported')
        else:
            score = scorer(averaged_metrics)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores


model_selection._validation._multimetric_score = _multimetric_score


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups):
        if isinstance(groups, list):
            groups = np.array(groups)
        elif not isinstance(groups, np.ndarray):
            raise RuntimeError('groups has not array like type')
        train = np.where(groups == 0)[0]

        test_ind = groups == 1
        if np.sum(test_ind) == 0:
            test = [0]
        else:
            test = np.where(groups == 1)[0]
        print(f'train split {train}')
        print(f'test split {test}')
        yield (train, test)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def _score(averaged_metrics, metric):
    average_score = averaged_metrics[metric]
    return average_score


def wrap_score(metric):
    partial_score = partial(_score, metric=metric)
    update_wrapper(partial_score, _score)
    return partial_score


def random_search(X, y, groups, param_distributions, rpe_indices, **kwargs):
    scoring = {metric: wrap_score(metric) for metric in ('ATE', 'RMSE_t', 'RMSE_r', 'RPE_t', 'RPE_r')}

    print(f'Number of predicted trajectories {len(X)}')
    print(f'Number of gt trajectories {len(y)}')
    print(f'Number of train trajectories {len(groups) - sum(groups)}')
    print(f'Number of test trajectories {sum(groups)}')
    print(f'RPE indices: {rpe_indices}')

    rs = RandomizedSearchCV(G2OEstimator(verbose=True, rpe_indices=rpe_indices),
                            param_distributions,
                            cv=DisabledCV(),
                            refit=False,
                            scoring=scoring,
                            **kwargs)
    rs.fit(X, y, groups)

    return pd.DataFrame(rs.cv_results_)
