import __init_path__
import env

import os
import unittest
import numpy as np
import pandas as pd

import env
from slam.aggregation import DummyAverager
from slam.evaluation import calculate_metrics, normalize_metrics
from slam.linalg import RelativeTrajectory

from slam.utils import visualize_trajectory_with_gt


class BaseTest(object):
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

    @staticmethod
    def evaluate(gt_trajectory, predicted_trajectory, file_name):
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}' for key, value in record.items()])
        title = f'{"03".upper()}: {trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory,
                                     predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'tests', f'{file_name}.html'),
                                     title=title)

        return record

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
            predict['t_z'] = np.random.normal(predict['t_z'], predict['t_z_confidence'])

        return predict

    def test_1(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        predict = self.gt2predict(gt)
        self.algorithm.append(predict)
        predicted_trajectory = self.algorithm.get_trajectory()
        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        self.evaluate(gt_trajectory, predicted_trajectory, 'test_1')

    def test_2(self):

        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv']

        gt_trajectory = None

        for index, p in enumerate(csv_paths):
            gt = pd.read_csv(os.path.join(env.PROJECT_PATH, p))
            prediction = self.gt2predict(gt)
            self.algorithm.append(prediction)

            if index == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        predicted_trajectory = self.algorithm.get_trajectory()

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_2')
        self.assert_almost_zero(record)

    def test_3(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv']

        gt_trajectory = None

        for index, p in enumerate(csv_paths):
            gt = pd.read_csv(os.path.join(env.PROJECT_PATH, p))
            prediction = self.gt2predict(gt)
            self.algorithm.append(prediction)

            if index == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        predicted_trajectory = self.algorithm.get_trajectory()

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_2')
        self.assert_almost_zero(record)

    def test_4(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv']

        gt_trajectory = None

        for index, p in enumerate(csv_paths):
            gt = pd.read_csv(os.path.join(env.PROJECT_PATH, p))
            prediction = self.gt2predict(gt)
            self.algorithm.append(prediction)

            if index == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        predicted_trajectory = self.algorithm.get_trajectory()

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_2')
        self.assert_almost_zero(record)

    def test_5(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()

        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_slam_predict = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt_slam_predict, noise=True)

        is_adjustment_measurements = (pred.to_index - pred.from_index) == 1
        adjustment_measurements = pred[is_adjustment_measurements].reset_index(drop=True)
        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory, 'test_5_noised')

        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory, 'test_5_optimized')

        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)


class TestDummyAverager(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        self.algorithm = DummyAverager()
