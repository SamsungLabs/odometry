import os
import unittest
import numpy as np
import pandas as pd

import __init_path__
import env

from slam.evaluation import calculate_metrics, normalize_metrics
from slam.linalg import RelativeTrajectory

from slam.aggregation.graph_optimizer import GraphOptimizer
from slam.aggregation import DummyAverage
from slam.utils import visualize_trajectory_with_gt


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.algorithm = None

    def assert_almost_zero(self, record):
        translation_precision = 10
        rotation_precision = 4
        self.assertAlmostEqual(record['ATE'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RPE_r'], 0, places=rotation_precision)
        self.assertAlmostEqual(record['RPE_t'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RMSE_t'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RMSE_r'], 0, places=rotation_precision)

    def assert_greater(self, record1, record2):
        self.assertGreater(record1['ATE'], record2['ATE'])
        self.assertGreater(record1['RPE_r'], record2['RPE_r'])
        self.assertGreater(record1['RPE_t'], record2['RPE_t'])
        self.assertGreater(record1['RMSE_t'], record2['RMSE_t'])
        self.assertGreater(record1['RMSE_r'], record2['RMSE_r'])

    def gt2predict(self, gt, noise=False):
        columns_mean = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        predict = gt[columns_mean]
        columns_std = [c + '_confidence' for c in columns_mean]

        predict[columns_std] = pd.DataFrame([[0.001] * len(columns_std)] * len(gt), index=predict.index)
        predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(x[4:-4]))
        predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(x[4:-4]))

        if noise:
            predict['euler_x'] = np.random.normal(predict['euler_x'], predict['euler_x_confidence'])
            predict['euler_y'] = np.random.normal(predict['euler_y'], predict['euler_y_confidence'])
            predict['euler_z'] = np.random.normal(predict['euler_z'], predict['euler_z_confidence'])
            predict['t_x'] = np.random.normal(predict['t_x'], predict['t_x_confidence'])
            predict['t_y'] = np.random.normal(predict['t_y'], predict['t_y_confidence'])
            # predict['t_z'] = np.random.normal(predict['t_z'], predict['t_z_confidence'])

        return predict

    def test_1(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt)
        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record.items()])
        title = f'{"03".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_1.html'), title=title)

        self.assert_almost_zero(record)

    def test_2(self):
        csv_path_s1 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv'
        gt_s1 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s1))
        pred_s1 = self.gt2predict(gt_s1)
        self.algorithm.append(pred_s1)

        csv_path_s2 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv'
        gt_s2 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s2))
        pred_s2 = self.gt2predict(gt_s2)
        self.algorithm.append(pred_s2)

        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_s1).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record.items()])
        title = f'{"03".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_2.html'), title=title)

        self.assert_almost_zero(record)

    def test_3(self):
        csv_path_s1 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_s1 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s1))
        pred_s1 = self.gt2predict(gt_s1)
        self.algorithm.append(pred_s1)

        csv_path_s2 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv'
        gt_s2 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s2))
        pred_s2 = self.gt2predict(gt_s2)
        self.algorithm.append(pred_s2)

        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_s1).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record.items()])
        title = f'{"00".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_3.html'), title=title)

        self.assert_almost_zero(record)

    def test_4(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))

        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_slam_predict = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt_slam_predict)
        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record.items()])
        title = f'{"00".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_4.html'), title=title)

        self.assert_almost_zero(record)

    def test_5(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))

        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_slam_predict = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt_slam_predict, noise=True)
        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        is_adjustment_measurements = (pred.to_index - pred.from_index) == 1
        adjustment_measurements = pred[is_adjustment_measurements].reset_index(drop=True)
        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()

        record_noised = calculate_metrics(gt_trajectory, noised_trajectory, rpe_indices='full')
        record_noised = normalize_metrics(record_noised)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record_noised.items()])
        title = f'{"00".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=noised_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_5_noised.html'), title=title)

        print('metrics before optimization', record_noised)

        record_optimized = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record_optimized = normalize_metrics(record_optimized)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                               for key, value in record_optimized.items()])
        title = f'{"00".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory, predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'test_5_optimized.html'), title=title)

        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)


class TestDummyAverage(BaseTest):
    def setUp(self) -> None:
        self.algorithm = DummyAverage()
