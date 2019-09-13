import __init_path__
import env

import os
import unittest
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path

import env
from slam.aggregation import DummyAverager
from slam.aggregation import GraphOptimizer
from slam.evaluation import calculate_metrics, normalize_metrics
from slam.linalg import RelativeTrajectory

from slam.utils import visualize_trajectory_with_gt


class BaseTest(object):
    def __init__(self):
        self.algorithm = None
        self.mean_cols = None
        self.std_cols = None
        self.draw_intermediate = None

    def set_up(self) -> None:
        self.algorithm = None
        self.mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.std_cols = [c + '_confidence' for c in self.mean_cols]
        self.draw_intermediate = False

    def assertAlmostEqual(self, in_1, in_2, places=None):
        raise RuntimeError('Not implemented')

    def assertGreater(self, in_1, in_2):
        raise RuntimeError('Not implemented')

    @staticmethod
    def read_csv(csv_path):
        return pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))

    def assert_almost_zero(self, record):
        translation_precision = 8
        rotation_precision = 4
        self.assertAlmostEqual(record['ATE'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RPE_r'], 0, places=rotation_precision)
        self.assertAlmostEqual(record['RPE_t'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RMSE_t'], 0, places=translation_precision)
        self.assertAlmostEqual(record['RMSE_r'], 0, places=rotation_precision)

    def assert_greater(self, record1, record2):
        for k in record1.keys():
            self.assertGreater(record1[k], record2[k])

    def predict(self, csv_paths):
        for p in csv_paths:
            prediction = self.df2slam_predict(self.read_csv(p))
            self.algorithm.append(prediction)

        return self.algorithm.get_trajectory()

    @staticmethod
    def calculate_metrics(gt_trajectory, predicted_trajectory, file_name):
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')
        record = normalize_metrics(record)

        trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}' for key, value in record.items()])
        title = f'{trajectory_metrics_as_str}'

        visualize_trajectory_with_gt(gt_trajectory=gt_trajectory,
                                     predicted_trajectory=predicted_trajectory,
                                     file_path=os.path.join(env.PROJECT_PATH, 'tests', f'{file_name}.html'),
                                     title=title)
        return record

    def df2slam_predict(self, gt):
        predict = gt[self.mean_cols]

        for std_col in self.std_cols:
            if std_col not in gt.columns:
                predict[std_col] = 1
            else:
                predict.loc[gt.index, std_col] = gt[std_col]

        predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(Path(x).stem))
        predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(Path(x).stem))
        return predict

    def get_noised_trajectory(self, df):
        for mean_col, std_col in zip(self.mean_cols, self.std_cols):
            df[std_col] = 0.001
            df[mean_col] = np.random.normal(df[mean_col], df[std_col])
        return df

    @staticmethod
    def get_odometry_trajectory(df, length=-1):

        if length != -1:
            df = df[:length]

        adjustment_measurements = BaseTest.get_adjustment_measurements(df)
        odometry_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        return odometry_trajectory

    @staticmethod
    def get_adjustment_measurements(df):
        is_adjacent = np.abs(df.to_index - df.from_index) <= 1
        measurements = df[is_adjacent].reset_index(drop=True)
        return measurements

    @staticmethod
    def get_adjustment_measurements_with_loops(df, gt=None):
        is_adjacent = np.abs(df.to_index - df.from_index) <= 1
        is_loop = (df.to_index - df.from_index) > 100

        if gt is not None:
            gt_loops = gt[is_loop]
            df.loc[gt_loops.index] = gt_loops

        measurements = df[is_adjacent | is_loop].reset_index(drop=True)
        return measurements

    def evaluate(self, gt, gt_trajectory, predict, prefix):
        for index in trange(1, len(gt_trajectory.positions) + 1):
            matches = predict[predict.to_index == index]
            self.algorithm.append(matches)

            if not self.draw_intermediate:
                continue

            if index % 100 == 0:
                gt_trajectory_partial = self.get_odometry_trajectory(gt, length=index)
                odometry_trajectory = self.get_odometry_trajectory(predict, length=index)
                self.calculate_metrics(gt_trajectory_partial, odometry_trajectory, f'{prefix}_odometry_i_{index}')
                predicted_trajectory = self.algorithm.get_trajectory()
                self.calculate_metrics(gt_trajectory, predicted_trajectory, f'{prefix}_slam_optimized_i_{index}')

        odometry_trajectory = self.get_odometry_trajectory(predict)
        odometry_record = self.calculate_metrics(gt_trajectory, odometry_trajectory, prefix + '_odometry')
        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.calculate_metrics(gt_trajectory, predicted_trajectory, prefix + f'_slam')
        print(' ')
        print('metrics before optimization', odometry_record)
        print('metrics after optimization ', record_optimized)
        self.assert_greater(odometry_record, record_optimized)

    def test_gt_df(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.calculate_metrics(gt_trajectory, predicted_trajectory, 'test_gt_df')
        self.assert_almost_zero(record)

    def test_gt_df_with_strides_1(self):

        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.calculate_metrics(gt_trajectory, predicted_trajectory, 'test_gt_df_with_strides_1')
        self.assert_almost_zero(record)

    def test_gt_df_with_strides_2(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.calculate_metrics(gt_trajectory, predicted_trajectory, 'test_gt_df_with_strides_2')
        self.assert_almost_zero(record)

    def test_gt_df_with_all_matches(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.calculate_metrics(gt_trajectory, predicted_trajectory, 'test_gt_df_with_all_matches')
        self.assert_almost_zero(record)

    def test_noised_df_with_all_matches(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_trajectory = self.get_odometry_trajectory(self.df2slam_predict(self.read_csv(csv_path_gt)))

        csv_path_noised = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_noised.csv'
        if not os.path.exists(os.path.join(env.PROJECT_PATH, csv_path_noised)):
            print('Generating new noisy trajectory')
            noised_df = self.get_noised_trajectory(self.read_csv(csv_path_gt))
            noised_df.to_csv(os.path.join(env.PROJECT_PATH, csv_path_noised))
        else:
            noised_df = self.read_csv(csv_path_noised)

        pred = self.df2slam_predict(noised_df)

        odometry_trajectory = self.get_odometry_trajectory(pred)
        odometry_record = self.calculate_metrics(gt_trajectory, odometry_trajectory,
                                                 'test_noised_df_with_all_matches_odometry')

        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()
        slam_record = self.calculate_metrics(gt_trajectory, predicted_trajectory,
                                             'test_noised_df_with_all_matches_slam')

        print(' ')
        print('metrics before optimization', odometry_record)
        print('metrics after optimization ', slam_record)

        self.assert_greater(odometry_record, slam_record)

    def test_square(self):
        return True
        gt_df = self.df2slam_predict(self.read_csv('tests/minidataset/toy/square_loop_gt.csv'))
        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()

        predicted_df = self.df2slam_predict(self.read_csv('tests/minidataset/toy/square_loop_predict.csv'))
        odometry_trajectory = self.get_odometry_trajectory(predicted_df)
        odometry_record = self.calculate_metrics(gt_trajectory, odometry_trajectory, f'test_square_loop_odometry')

        self.algorithm.append(predicted_df)
        predicted_trajectory = self.algorithm.get_trajectory()
        slam_record = self.calculate_metrics(gt_trajectory, predicted_trajectory, f'test_square_loop_slam')

        print(' ')
        print('metrics before optimization', odometry_record)
        print('metrics after optimization ', slam_record)

        self.assert_greater(odometry_record, slam_record)

    def test_predict_with_gt_loops_only(self):
        return True
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt = self.df2slam_predict(self.read_csv(csv_path_gt))
        gt_trajectory = self.get_odometry_trajectory(gt)

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        predict = self.read_csv(csv_path_real_predict)
        predict = self.get_adjustment_measurements_with_loops(predict, gt)

        self.evaluate(gt, gt_trajectory, predict, prefix='test_predict_with_gt_loops_only')

    def test_predict_with_predicted_loops_only(self):
        return True
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt = self.df2slam_predict(self.read_csv(csv_path_gt))
        gt_trajectory = self.get_odometry_trajectory(gt)

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        predict = self.read_csv(csv_path_real_predict)
        predict = self.get_adjustment_measurements_with_loops(predict)

        self.evaluate(gt, gt_trajectory, predict, prefix='test_predict_with_predicted_loops_only')


class TestDummyAverager(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        super().set_up()
        self.algorithm = DummyAverager()
        self.draw_intermediate = False


class TestGraphOptimizer(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        super().set_up()
        self.algorithm = GraphOptimizer(max_iterations=5000, online=False, verbose=True)
        self.draw_intermediate = False
