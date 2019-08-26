import __init_path__
import env

import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import env
from slam.aggregation import DummyAverager
from slam.aggregation import GraphOptimizer
from slam.evaluation import calculate_metrics, normalize_metrics
from slam.linalg import RelativeTrajectory

from slam.utils import visualize_trajectory_with_gt


class BaseTest(object):

    def set_up(self) -> None:
        self.algorithm = None
        self.mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.std_cols = [c + '_confidence' for c in self.mean_cols]

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
        for k in record1.keys():
            self.assertGreater(record1[k], record2[k])

    def predict(self, csv_paths):
        for p in csv_paths:
            prediction = self.df2slam_predict(self.read_csv(p))
            self.algorithm.append(prediction)

        return self.algorithm.get_trajectory()

    @staticmethod
    def evaluate(gt_trajectory, predicted_trajectory, file_name):
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

        predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(Path(x).stem))
        predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(Path(x).stem))
        return predict

    def generate_noised_trajectory(self, df):
        for mean_col, std_col in zip(self.mean_cols, self.std_cols):
            df[std_col] = 0.001
            df[mean_col] = np.random.normal(df[mean_col], df[std_col])
        return df

    # def test_1(self):
    #     csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv']
    #
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
    #     predicted_trajectory = self.predict(csv_paths)
    #
    #     record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_1')
    #     self.assert_almost_zero(record)
    #
    # def test_2(self):
    #
    #     csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv',
    #                  'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv']
    #
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
    #     predicted_trajectory = self.predict(csv_paths)
    #
    #     record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_2')
    #     self.assert_almost_zero(record)
    #
    # def test_3(self):
    #     csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
    #                  'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv']
    #
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
    #     predicted_trajectory = self.predict(csv_paths)
    #
    #     record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_3')
    #     self.assert_almost_zero(record)
    #
    # def test_4(self):
    #     csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
    #     csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv']
    #
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()
    #     predicted_trajectory = self.predict(csv_paths)
    #
    #     record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_4')
    #     self.assert_almost_zero(record)
    #
    # def test_5(self):
    #     csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()
    #
    #     csv_path_noised = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_noised.csv'
    #     csv_path_mixed = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
    #     if not os.path.exists(os.path.join(env.PROJECT_PATH, csv_path_noised)):
    #         print('Generating new noisy trajectory')
    #         noised_df = self.generate_noised_trajectory(self.read_csv(csv_path_mixed))
    #         noised_df.to_csv(os.path.join(env.PROJECT_PATH, csv_path_noised))
    #     else:
    #         noised_df = self.read_csv(csv_path_noised)
    #
    #     pred = self.df2slam_predict(noised_df)
    #
    #     is_adjustment_measurements = (pred.to_index - pred.from_index) == 1
    #     adjustment_measurements = pred[is_adjustment_measurements].reset_index(drop=True)
    #     noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
    #     record_noised = self.evaluate(gt_trajectory, noised_trajectory, 'test_5_noised')
    #
    #     self.algorithm.append(pred)
    #     predicted_trajectory = self.algorithm.get_trajectory()
    #     record_optimized = self.evaluate(gt_trajectory, predicted_trajectory, 'test_5_optimized')
    #
    #     print('metrics before optimization', record_noised)
    #     print('metrics after optimization ', record_optimized)
    #
    #     self.assert_greater(record_noised, record_optimized)

    def test_real_predict_with_loops_only(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_df = self.read_csv(csv_path_gt)
        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()

        csv_path_gt_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_pred = self.read_csv(csv_path_gt_predict)

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        real_predict = self.read_csv(csv_path_real_predict)

        is_adjustment_measurements = (real_predict.to_index - real_predict.from_index) <= 1
        adjustment_measurements = real_predict[is_adjustment_measurements].reset_index(drop=True)
        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory, 'test_real_predict_with_loops_only_odometry')

        is_loop_closure = (real_predict.to_index - real_predict.from_index) > 100
        adjustment_measurements_with_loop_closure = real_predict[is_adjustment_measurements | is_loop_closure].reset_index(drop=True)

        for index in range(1, len(adjustment_measurements)):
            matches = adjustment_measurements_with_loop_closure[adjustment_measurements_with_loop_closure.to_index == index]
            self.algorithm.append(matches)

            is_loop_closure = np.any((matches.to_index - matches.from_index) > 100)

            if index % 100 == 0 or is_loop_closure:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt_df[:index]).to_global()

                noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements[:index]).to_global()
                record_noised = self.evaluate(gt_trajectory, noised_trajectory,
                                              f'test_real_predict_with_loops_only_slam_noised_i_{index}')

                predicted_trajectory = self.algorithm.get_trajectory()
                record_optimized = self.evaluate(gt_trajectory, predicted_trajectory,
                                                 f'test_real_predict_with_loops_only_slam_optimized_i_{index}')

                print(f'Saved. Index={index}')

                if is_loop_closure:
                    break

        print(' ')
        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)

    # def test_real_predict(self):
    #     csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
    #     gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()
    #
    #     csv_path_noised = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
    #     pred = self.read_csv(csv_path_noised)
    #
    #     is_adjustment_measurements = (pred.to_index - pred.from_index) == 1
    #     adjustment_measurements = pred[is_adjustment_measurements].reset_index(drop=True)
    #     noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
    #     record_noised = self.evaluate(gt_trajectory, noised_trajectory, 'test_real_predict')
    #
    #     self.algorithm.append(pred)
    #     predicted_trajectory = self.algorithm.get_trajectory()
    #     record_optimized = self.evaluate(gt_trajectory, predicted_trajectory, 'test_real_predict')
    #
    #     print(' ')
    #     print('metrics before optimization', record_noised)
    #     print('metrics after optimization ', record_optimized)
    #
    #     self.assert_greater(record_noised, record_optimized)


class TestDummyAverager(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        super().set_up()
        self.algorithm = DummyAverager()


class TestGraphOptimizer(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        super().set_up()
        self.algorithm = GraphOptimizer(2000)
