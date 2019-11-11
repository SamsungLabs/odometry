import numbers
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from functools import partial, update_wrapper

import __init_path__
import env

from scripts.aggregation.base_search import Search, DisabledCV
from slam.aggregation import G2OEstimator


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


class RandomSearch(Search):

    def _score(self, averaged_metrics, metric):
        average_score = averaged_metrics[metric]
        return average_score

    def wrap_score(self, metric):
        partial_score = partial(self._score, metric=metric)
        update_wrapper(partial_score, self._score)
        return partial_score

    def search(self,
               X,
               y,
               groups,
               param_distributions,
               rpe_indices,
               n_iter,
               n_jobs=3,
               verbose=True):
        scoring = {metric: self.wrap_score(metric) for metric in ('ATE', 'RMSE_t', 'RMSE_r', 'RPE_t', 'RPE_r')}

        print(f'Number of predicted trajectories {len(X)}')
        print(f'Number of gt trajectories {len(y)}')
        print(f'Number of train trajectories {len(groups) - sum(groups)}')
        print(f'Number of test trajectories {sum(groups)}')
        print(f'RPE indices: {rpe_indices}')

        rs = RandomizedSearchCV(G2OEstimator(verbose=True, rpe_indices=rpe_indices),
                                param_distributions,
                                cv=DisabledCV(),
                                refit=False,
                                scoring=scoring)
        rs.fit(X, y, groups)

        return pd.DataFrame(rs.cv_results_)


if __name__ == '__main__':
    parser = RandomSearch.get_default_parser()
    args = parser.parse_args()
    RandomSearch().start(**vars(args))
