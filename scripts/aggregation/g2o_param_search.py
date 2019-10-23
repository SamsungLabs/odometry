import time
import sklearn
from sklearn import model_selection
import numpy as np
from functools import partial, update_wrapper
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import numbers

from .g2o_estimator import G2OEstimator
from slam.evaluation import calculate_metrics, normalize_metrics


def _multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring.
       Original functions calculated predic for every score. This function evaluate predict only one time"""
    print('Redeclaring __multimetric_score function')
    preds = estimator.predict(X_test)
    
    scores = {}    
    for name, scorer in scorers.items():
        if y_test is None:
            raise RuntimeError('Not supported')
        else:
            score = scorer(y_test, preds)

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
        elif not isinstance(groups, nd.array):
            raise RuntimeError(f'groups has not array like type') 
        train = np.where(groups == 0)[0]
        test = np.where(groups == 1)[0]
        print(f'train split {train}')
        print(f'test split {test}')
        yield (train, test)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def _score(y, preds, metric):
    print(f'Scoring {len(y)} trajectories. Metric: {metric}')
    start_time = time.time()
    scores = []
    for i, (gt_trajectory, predicted_trajectory) in enumerate(zip(y, preds)):
        metrics_dict = calculate_metrics(gt_trajectory, predicted_trajectory)
        metrics_dict = normalize_metrics(metrics_dict)
        scores.append(metrics_dict[metric])
    
    average_score = np.mean(scores)
    print(f'Scoring completed in {time.time() - start_time:.3f} s\n') 
    return average_score


def wrap_score(metric):
    partial_score = partial(_score, metric=metric)
    update_wrapper(partial_score, _score)
    return partial_score


def random_search(X, y, groups, param_distributions, **kwargs):
    
#     scoring = {metric: make_scorer(score(metric), greater_is_better=False) 
#                for metric in ('ATE', 'RMSE_t', 'RMSE_r', 'RPE_t', 'RPE_r')}

    scoring = {metric: wrap_score(metric) for metric in ('ATE', 'RMSE_t', 'RMSE_r', 'RPE_t', 'RPE_r')}
    
    
    print(f'Number of predicted trajectories {len(X)}')
    print(f'Number of gt trajectories {len(y)}')
    print(f'Number of train trajectories {len(groups) - sum(groups)}')
    print(f'Number of test trajectories {sum(groups)}')

    rs = RandomizedSearchCV(G2OEstimator(verbose=True), 
                            param_distributions,
                            cv=DisabledCV(),
                            refit=False,
                            scoring=scoring,
                            **kwargs)
    rs.fit(X, y, groups)

    return pd.DataFrame(rs.cv_results_)
