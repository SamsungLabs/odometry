import time
import numpy as np
from sklearn.base import BaseEstimator

import __init_path__
import env

from slam.aggregation import GraphOptimizer
from slam.evaluation import calculate_metrics


class G2OEstimator(BaseEstimator):

    def __init__(self,
                 coef={1: 0},
                 coef_loop=0,
                 loop_threshold=0,
                 rotation_scale=1,
                 max_iterations=100,
                 online=False,
                 verbose=False):
        self.coef = coef
        self.coef_loop = coef_loop
        self.loop_threshold = loop_threshold
        self.rotation_scale = rotation_scale
        self.max_iterations = max_iterations
        self.online = online
        self.verbose = verbose

    @property
    def mean_cols(self):
        return ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']

    @property
    def std_cols(self):
        return [c + '_confidence' for c in self.mean_cols]

    @property
    def all_cols(self):
        return ['from_index', 'to_index'] + self.mean_cols + self.std_cols

    def _apply_g2o_coef(self, row):
        diff = row['diff']

        std_coef = 1
        if diff in self.coef:
            std_coef = self.coef[diff]
        else:
            is_loop = diff > self.loop_threshold
            std_coef = self.coef_loop if is_loop else 1e7

        row[self.std_cols] *= std_coef
        row[['euler_x_confidence', 'euler_y_confidence', 'euler_z_confidence']] *= self.rotation_scale
        return row 

    def fit(self, X, y, sample_weight=None):
        print(f'Running {self}\n')

    def predict(self, X):
        if self.verbose:
            start_time = time.time()
            print(f'Predicting for {len(X)} trajectories...')

        preds = []
        for df in X:
            df_with_coef = df.apply(self._apply_g2o_coef, axis=1)

            g2o = GraphOptimizer(max_iterations=self.max_iterations, online=self.online)
            g2o.append(df_with_coef[self.all_cols])
            predicted_trajectory = g2o.get_trajectory()
            preds.append(predicted_trajectory)

        if self.verbose:
            print(f'Predicting completed in {time.time() - start_time:.3f} s\n') 
        return preds

    def score(self, X, y, sample_weight=None):
        preds = self.predict(X)

        if self.verbose:
            start_time = time.time()
            print(f'Scoring {len(X)} trajectories...')

        scores = []
        for i, (gt_trajectory, predicted_trajectory) in enumerate(zip(y, preds)):
            metrics_dict = calculate_metrics(gt_trajectory, predicted_trajectory)

            score = metrics_dict['ATE']
            print(f'\t{i + 1}: {score}')
            scores.append(score)

        if sample_weight is not None:
            scores = [score * w for score, w in zip(scores, sample_weight)]

        average_score = np.mean(scores)

        if self.verbose:
            print(f'Scoring completed in {time.time() - start_time:.3f} s')
            print(f'Average score: {average_score}\n')

        return average_score
