import numpy as np
import pandas as pd

import __init_path__
import env

from scripts.aggregation.base_search import Search, DisabledCV
from slam.aggregation import G2OEstimator


class GridSearch(Search):
    @staticmethod
    def get_default_parser():
        parser = Search.get_default_parser()
        parser.add_argument('rank_metric', type=str, choices=['ATE', 'RPE'])
        return parser

    def log_predict(self, estimator, X_val, y_val, X_test, y_test):
        result = estimator.log_params()
        predict = estimator.predict(X_val, y_val)
        result = result.append(pd.DataFrame({'val_' + k: v for k, v in predict.items()}))
        predict = estimator.predict(X_test, y_test)
        result = result.append(pd.DataFrame({'test_' + k: v for k, v in predict.items()}))
        return result

    def search(self,
               X,
               y,
               groups,
               param_distributions,
               rpe_indices,
               n_iter,
               n_jobs=3,
               verbose=True,
               **kwargs):

        loop_coefs = param_distributions['coef_loop']
        stride_coefs = param_distributions['coef']

        assert len(loop_coefs) == 1 or len(stride_coefs) == 1

        val_ind, test_ind = DisabledCV().split(X, y, groups)
        X_val = [X[ind] for ind in val_ind]
        y_val = [y[ind] for ind in val_ind]
        X_test = [X[ind] for ind in test_ind]
        y_test = [y[ind] for ind in test_ind]

        result = pd.DataFrame()
        if isinstance(loop_coefs, list) and len(loop_coefs) > 1:
            for c in loop_coefs:
                for threshold in param_distributions['loop_threshold']:
                    estimator = G2OEstimator(coef=stride_coefs[0],
                                             coef_loop=c,
                                             loop_threshold=threshold,
                                             rotation_scale=param_distributions['rotation_scale'][0],
                                             max_iterations=param_distributions['max_iterations'][0],
                                             rpe_indices=rpe_indices,
                                             verbose=True
                                             )
                    result = result.append(self.log_predict(estimator, X_val, y_val, X_test, y_test))
        else:
            for coefs in param_distributions:
                estimator = G2OEstimator(coef=coefs,
                                         coef_loop= param_distributions['coef_loop'][0],
                                         loop_threshold=param_distributions['loop_threshold'][0],
                                         rotation_scale=param_distributions['rotation_scale'][0],
                                         max_iterations=param_distributions['max_iterations'][0],
                                         rpe_indices=rpe_indices,
                                         verbose=True)

                result = result.append(self.log_predict(estimator, X_val, y_val, X_test, y_test))

        key = 'val' + kwargs['rank_metric']
        best_run_ind = np.argmin(result[key].values)
        for rotation_scale in param_distributions['rotation_scale']:
            estimator = G2OEstimator(coef=result['coef'].values[best_run_ind],
                                     coef_loop=result['coef_loop'].values[best_run_ind],
                                     loop_threshold=result['loop_threshold'].values[best_run_ind],
                                     rotation_scale=rotation_scale,
                                     max_iterations=result['max_iterations'].values[best_run_ind],
                                     rpe_indices=rpe_indices,
                                     verbose=True)
            result = result.append(self.log_predict(estimator, X_val, y_val, X_test, y_test))

        return result


if __name__ == '__main__':
    parser = GridSearch.get_default_parser()
    args = parser.parse_args()
    GridSearch.start(**vars(args))
