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

    def set_up(self) -> None:
        self.algorithm = None
        self.columns_mean = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.columns_std = [c + '_confidence' for c in self.columns_mean]

    def read_csv(self, csv_path):
        return pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))

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

    def predict(self, csv_paths):
        for p in csv_paths:
            prediction = self.df2slam_predict(self.read_csv(p))
            self.algorithm.append(prediction)

        predicted_trajectory = self.algorithm.get_trajectory()
        return predicted_trajectory

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

    def df2slam_predict(self, gt):
        predict = gt[self.columns_mean]
        predict[self.columns_std] = pd.DataFrame([[1] * len(self.columns_std)] * len(gt), index=predict.index)
        predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(x[4:-4]))
        predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(x[4:-4]))
        return predict

    def generate_noised_trajectory(self, df):
        df[self.columns_std] = pd.DataFrame([[0.001] * len(self.columns_std)] * len(df), index=df.index)

        df['euler_x'] = np.random.normal(df['euler_x'], df['euler_x_confidence'])
        df['euler_y'] = np.random.normal(df['euler_y'], df['euler_y_confidence'])
        df['euler_z'] = np.random.normal(df['euler_z'], df['euler_z_confidence'])
        df['t_x'] = np.random.normal(df['t_x'], df['t_x_confidence'])
        df['t_y'] = np.random.normal(df['t_y'], df['t_y_confidence'])
        df['t_z'] = np.random.normal(df['t_z'], df['t_z_confidence'])

        return df

    def test_1(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_1')
        self.assert_almost_zero(record)

    def test_2(self):

        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_2')
        self.assert_almost_zero(record)

    def test_3(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_3')
        self.assert_almost_zero(record)

    def test_4(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_4')
        self.assert_almost_zero(record)

    def test_5(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()

        csv_path_noised = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixesd_noised.csv'
        csv_path_mixed = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        if not os.path.exists(os.path.join(env.PROJECT_PATH, csv_path_noised)):
            noised_df = self.generate_noised_trajectory(self.read_csv(csv_path_mixed))
            noised_df.to_csv(os.path.join(env.PROJECT_PATH, csv_path_noised))
        else:
            noised_df = self.read_csv(csv_path_noised)

        pred = self.df2slam_predict(noised_df)

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
        super().set_up()
        self.algorithm = DummyAverager()
