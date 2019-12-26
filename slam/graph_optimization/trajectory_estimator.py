import os
import time

from slam.graph_optimization import GraphOptimizer
from slam.utils import visualize_trajectory_with_gt
from slam.evaluation import calculate_metrics, normalize_metrics, average_metrics


class TrajectoryEstimator:

    def __init__(self,
                 strides_sigmas,
                 loop_sigma=0,
                 loop_threshold=0,
                 rotation_weight=1,
                 max_iterations=100,
                 online=False,
                 verbose=False,
                 rpe_indices='full',
                 vis_dir=None,
                 pred_dir=None):
        self.strides_sigmas = strides_sigmas
        self.loop_sigma = loop_sigma
        self.loop_threshold = loop_threshold
        self.rotation_weight = rotation_weight
        self.max_iterations = max_iterations
        self.online = online
        self.verbose = verbose
        self.rpe_indices = rpe_indices

        if vis_dir is not None:
            self.vis_dir = vis_dir
            if not os.path.isdir(self.vis_dir):
                os.mkdir(self.vis_dir)

        if pred_dir is not None:
            self.pred_dir = pred_dir
            if not os.path.isdir(self.pred_dir):
                os.mkdir(self.pred_dir)

    @property
    def mean_cols(self):
        return ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']

    @property
    def std_cols(self):
        return [c + '_confidence' for c in self.mean_cols]

    @property
    def all_cols(self):
        return ['from_index', 'to_index'] + self.mean_cols + self.std_cols

    def log_params(self):
        params = {'coef': [self.strides_sigmas],
                  'coef_loop': [self.loop_sigma],
                  'loop_threshold': [self.loop_threshold],
                  'rotation_scale': [self.rotation_weight],
                  'max_iterations': [self.max_iterations]}
        return params

    def _apply_g2o_coef(self, row):
        diff = row['diff']

        if diff in self.strides_sigmas:
            std_coef = self.strides_sigmas[diff]
        else:
            is_loop = diff > self.loop_threshold
            std_coef = self.loop_sigma if is_loop else 1e15

        row[self.std_cols] *= std_coef
        row[['euler_x_confidence', 'euler_y_confidence', 'euler_z_confidence']] *= self.rotation_weight
        return row

    def predict(self, X, y, visualize=False, trajectory_names=None):
        if self.verbose:
            start_time = time.time()
            print(f'Predicting for {len(X)} trajectories...')

        preds = []
        for i, df in enumerate(X):
            consecutive_ind = df['diff'] == 1
            print(f'\t{i + 1}. Len {len(df[consecutive_ind])}')
            df_with_coef = df.apply(self._apply_g2o_coef, axis=1)

            g2o = GraphOptimizer(max_iterations=self.max_iterations, online=self.online)
            g2o.append(df_with_coef[self.all_cols])
            predicted_trajectory = g2o.get_trajectory()
            preds.append(predicted_trajectory)

        records = list()
        for i, (gt_trajectory, predicted_trajectory) in enumerate(zip(y, preds)):
            record = calculate_metrics(gt_trajectory, predicted_trajectory, self.rpe_indices)
            if visualize:
                trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}' for key, value in record.items()])
                if trajectory_names is not None:
                    file_path = os.path.join(self.vis_dir, f'{trajectory_names[i]}.html')
                else:
                    file_path = os.path.join(self.vis_dir, f'{i}.html')
                visualize_trajectory_with_gt(gt_trajectory=gt_trajectory,
                                             predicted_trajectory=predicted_trajectory,
                                             file_path=file_path,
                                             title=trajectory_metrics_as_str)

            if visualize and self.pred_dir is not None:
                predicted_trajectory.to_dataframe().to_csv(os.path.join(self.pred_dir, f'{trajectory_names[i]}.csv'))

            print(f'Trajectory len: {len(gt_trajectory)}')
            for k, v in normalize_metrics(record).items():
                print(f'>>>{k}: {v}')

            records.append(record)

        averaged_metrics = average_metrics(records)

        if self.verbose:
            print(f'Predicting completed in {time.time() - start_time:.3f} s\n')
        return averaged_metrics
