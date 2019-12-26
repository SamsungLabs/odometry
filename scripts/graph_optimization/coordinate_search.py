import numpy as np
import pandas as pd

import __init_path__
import env

from scripts.graph_optimization.base_search import Search, DisabledCV
from slam.graph_optimization import TrajectoryEstimator


class GridSearch(Search):
    """
    This class optimizes g2o parameters on validation trajectories.

    Input:
        predicted csv files from networks trained on different strides. I.e. 1_df.csv  and loops_df.csv --
        predictions from networks trained on stride 1 and results of relocalization estimators.

    In class implemented strategy of coordinate optimization. Algorithm:
    1. Pick reference prediction (defined by "best_stride" arg) and initialize graph
    2. Add prediction from networks train on relocalization estimator (loops_df.csv in example) to graph and optimize
     weight for that prediction that leads to the best metric (defined by 'rank_metric' arg)
    3. For every other prediction add it to graph and optimize weight for that prediction.
    4. Optimize rotation weight in graph constraints.
    """
    def __init__(self, vis_dir,
                 rank_metric,
                 best_stride,
                 **kwargs):
        self.vis_dir = vis_dir
        self.rank_column = f'val_{rank_metric}'
        self.best_stride = best_stride
        self.rpe_indices = None
        self.strides = None
        self.rpe_indices = None

    @staticmethod
    def get_default_parser():
        parser = Search.get_default_parser()
        parser.add_argument('--rank_metric', type=str, choices=['ATE', 'RPE'])
        parser.add_argument('--vis_dir', type=str)
        parser.add_argument('--best_stride', type=int, default=1)
        return parser

    def log_predict(self, estimator, X, y):
        params = estimator.log_params()
        val_predict = estimator.predict(X[0], y[0])
        val_predict = {'val_' + k: [v] for k, v in val_predict.items()}
        test_predict = estimator.predict(X[1], y[1])
        test_predict = {'test_' + k: [v] for k, v in test_predict.items()}
        result = pd.DataFrame({**params, **val_predict, **test_predict})
        result['val_RPE'] = result['val_RPE_r'] * 2 + result['val_RPE_r']
        return result

    def get_best_params(self, results):
        best_run_ind = np.argmin(results[self.rank_column].values)
        return dict(results.iloc[best_run_ind])

    def find_best_coef_loop(self, X, y, param_distributions, log):
        for c in self.get_coef_values():
            for threshold in param_distributions['loop_threshold']:
                estimator = TrajectoryEstimator(strides_std_weights={self.best_stride: 1},
                                                loop_std_weight=c,
                                                loop_threshold=threshold,
                                                rotation_weight=param_distributions['rotation_scale'][0],
                                                max_iterations=param_distributions['max_iterations'][0],
                                                rpe_indices=self.rpe_indices,
                                                verbose=True
                                                )
                log = log.append(self.log_predict(estimator, X, y))
        return log

    def find_best_coef(self, X, y, parent_log):
        best_params = self.get_best_params(parent_log)
        available_strides = list(set(self.strides) - set(best_params['coef'].keys()))
        if len(available_strides) == 0:
            return parent_log

        stride = min(available_strides)

        local_log = pd.DataFrame()
        for c in self.get_coef_values():
            best_params['coef'] = {**best_params['coef'], **{stride: c}}
            estimator = TrajectoryEstimator(**best_params, rpe_indices=self.rpe_indices, verbose=True)
            local_log = local_log.append(self.log_predict(estimator, X, y))

        child_log = self.find_best_coef(X, y, local_log)
        parent_log = parent_log.append(child_log)
        return parent_log

    def find_best_rotation_scale(self, X, y, param_distributions, log):
        best_params = self.get_best_params(log)
        for rotation_scale in param_distributions['rotation_scale'][1:]:
            best_params['rotation_scale'] = rotation_scale
            estimator = TrajectoryEstimator(**best_params, rpe_indices=self.rpe_indices, verbose=True)
            log = log.append(self.log_predict(estimator, X, y))
        return log

    def visualize(self, X, y, log, trajectory_names):
        best_params = self.get_best_params(log)
        estimator = TrajectoryEstimator(**best_params,
                                        rpe_indices=self.rpe_indices,
                                        verbose=True,
                                        vis_dir=self.vis_dir)

        estimator.predict(X, y, visualize=True, trajectory_names=trajectory_names)

    def search(self,
               X,
               y,
               groups,
               param_distributions,
               rpe_indices,
               trajectory_names=None,
               **kwargs):

        self.rpe_indices = rpe_indices
        self.strides = param_distributions['coef'][0].keys()

        val_ind, test_ind = next(DisabledCV().split(X, y, groups))
        X_split = ([X[ind] for ind in val_ind], [X[ind] for ind in test_ind])
        y_split = ([y[ind] for ind in val_ind], [y[ind] for ind in test_ind])

        log = pd.DataFrame()
        log = self.find_best_coef_loop(X_split, y_split, param_distributions, log)
        print(log)
        log = self.find_best_coef(X_split, y_split, log)
        print(log)
        log = self.find_best_rotation_scale(X_split, y_split, param_distributions, log)
        self.visualize(X, y, log, trajectory_names)
        return log


if __name__ == '__main__':
    parser = GridSearch.get_default_parser()
    args = parser.parse_args()
    GridSearch(args.vis_dir, args.rank_metric, args.best_stride).start(**vars(args))
