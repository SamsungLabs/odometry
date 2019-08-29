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

    def set_up(self) -> None:
        self.algorithm = None
        self.mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.std_cols = [c + '_confidence' for c in self.mean_cols]
        self.draw_intermediate = False

    def read_csv(self, csv_path):
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
            df[std_col] = 0.1
            df[mean_col] = np.random.normal(df[mean_col], df[std_col])
        return df

    def test_gt_df(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_gt_df')
        self.assert_almost_zero(record)

    def test_gt_df_with_strides_1(self):

        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_gt_df_with_strides_1')
        self.assert_almost_zero(record)

    def test_gt_df_with_strides_2(self):
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv',
                     'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_paths[0])).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_gt_df_with_strides_2')
        self.assert_almost_zero(record)

    def test_gt_df_with_all_matches(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        csv_paths = ['tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv']

        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()
        predicted_trajectory = self.predict(csv_paths)

        record = self.evaluate(gt_trajectory, predicted_trajectory, 'test_gt_df_with_all_matches')
        self.assert_almost_zero(record)

    def test_noised_df_with_all_matches(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_trajectory = RelativeTrajectory.from_dataframe(self.read_csv(csv_path_gt)).to_global()

        csv_path_noised = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_noised.csv'
        csv_path_mixed = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        if not os.path.exists(os.path.join(env.PROJECT_PATH, csv_path_noised)):
            print('Generating new noisy trajectory')
            noised_df = self.generate_noised_trajectory(self.read_csv(csv_path_mixed))
            noised_df.to_csv(os.path.join(env.PROJECT_PATH, csv_path_noised))
        else:
            noised_df = self.read_csv(csv_path_noised)

        pred = self.df2slam_predict(noised_df)

        is_adjustment_measurements = (pred.to_index - pred.from_index) == 1
        adjustment_measurements = pred[is_adjustment_measurements].reset_index(drop=True)
        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory, 'test_noised_df_with_all_matches_noised')

        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory,
                                         'test_noised_df_with_all_matches_optimized')

        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)

    def test_square(self):
        gt_df = pd.DataFrame()
        for i in range(299):
            gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                               't_x': [1], 't_y': [0], 't_z': [0],
                                               'from_index': [i], 'to_index': [i + 1]}))

        gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                           't_x': [1], 't_y': [0], 't_z': [0],
                                           'from_index': [299], 'to_index': [300]}))

        for i in range(300, 599):
            gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [00], 'euler_z': [0],
                                               't_x': [1], 't_y': [0], 't_z': [0],
                                               'from_index': [i], 'to_index': [i + 1]}))

        gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                           't_x': [1], 't_y': [0], 't_z': [0],
                                           'from_index': [599], 'to_index': [600]}))

        for i in range(600, 899):
            gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                               't_x': [1], 't_y': [0], 't_z': [0],
                                               'from_index': [i], 'to_index': [i + 1]}))

        gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                           't_x': [1], 't_y': [0], 't_z': [0],
                                           'from_index': [899], 'to_index': [900]}))

        for i in range(900, 1199):
            gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                               't_x': [1], 't_y': [0], 't_z': [0],
                                               'from_index': [i], 'to_index': [i + 1]}))

        gt_df = gt_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                           't_x': [1], 't_y': [0], 't_z': [0],
                                           'from_index': [1199], 'to_index': [1200]}))

        noised_df = pd.DataFrame()
        for i in range(299):
            noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                                       't_x': [1], 't_y': [0], 't_z': [0],
                                                       'from_index': [i], 'to_index': [i + 1]}))

        noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                                   't_x': [1], 't_y': [0], 't_z': [0],
                                                   'from_index': [299], 'to_index': [300]}))

        for i in range(300, 599):
            noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                                       't_x': [1], 't_y': [0], 't_z': [0.1],
                                                       'from_index': [i], 'to_index': [i + 1]}))

        noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                                   't_x': [1], 't_y': [0], 't_z': [0],
                                                   'from_index': [599], 'to_index': [600]}))

        for i in range(600, 899):
            noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                                       't_x': [1], 't_y': [0], 't_z': [0],
                                                       'from_index': [i], 'to_index': [i + 1]}))

        noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                                   't_x': [1], 't_y': [0], 't_z': [0],
                                                   'from_index': [899], 'to_index': [900]}))

        for i in range(900, 1199):
            noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                                       't_x': [1], 't_y': [0], 't_z': [0],
                                                       'from_index': [i], 'to_index': [i + 1]}))

        noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [-np.pi/2], 'euler_z': [0],
                                                   't_x': [1], 't_y': [0], 't_z': [0],
                                                   'from_index': [1199], 'to_index': [1200]}))

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
        noised_trajectory = RelativeTrajectory().from_dataframe(noised_df).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory, f'test_square_noised')

        noised_df = noised_df.append(pd.DataFrame({'euler_x': [0], 'euler_y': [0], 'euler_z': [0],
                                                   't_x': [0], 't_y': [0], 't_z': [0],
                                                   'from_index': [1200], 'to_index': [0]}))
        self.algorithm.append(noised_df)
        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory, f'test_square_optimized')

        print(' ')
        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)

    def test_real_predict_with_gt_loops_only(self):
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_df = self.read_csv(csv_path_gt)

        csv_path_gt_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed.csv'
        gt_pred = self.df2slam_predict(self.read_csv(csv_path_gt_predict))

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        real_predict = self.read_csv(csv_path_real_predict)

        is_adjustment_measurements = (real_predict.to_index - real_predict.from_index) <= 1
        adjustment_measurements = real_predict[is_adjustment_measurements].reset_index(drop=True)

        is_loop = (real_predict.to_index - real_predict.from_index) > 100
        gt_loops = gt_pred[is_loop]
        real_predict.loc[gt_loops.index] = gt_loops

        adjustment_measurements_with_loops = real_predict[is_adjustment_measurements | is_loop].reset_index(drop=True)

        for index in trange(1, len(adjustment_measurements) + 1):
            matches = adjustment_measurements_with_loops[adjustment_measurements_with_loops.to_index == index]
            self.algorithm.append(matches)

            if not self.draw_intermediate:
                continue

            if index % 100 == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt_df[:index]).to_global()

                noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements[:index]).to_global()
                self.evaluate(gt_trajectory, noised_trajectory,
                              f'test_real_predict_with_gt_loops_only_noised_i_{index}')

                predicted_trajectory = self.algorithm.get_trajectory()
                self.evaluate(gt_trajectory, predicted_trajectory,
                              f'test_real_predict_with_gt_loops_only_slam_optimized_i_{index}')

                print(f'Saved. Index={index}')

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()

        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory,
                                      f'test_real_predict_with_gt_loops_only_noised')

        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory,
                                         f'test_real_predict_with_gt_loops_only_slam_optimized')

        print(' ')
        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)

    def test_real_predict_with_predicted_loops_only(self):
        # Data for testing is not ready yet
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_df = self.read_csv(csv_path_gt)

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        real_predict = self.read_csv(csv_path_real_predict)

        is_adjustment_measurements = (real_predict.to_index - real_predict.from_index) <= 1
        adjustment_measurements = real_predict[is_adjustment_measurements].reset_index(drop=True)

        is_loop = (real_predict.to_index - real_predict.from_index) > 100
        adjustment_measurements_with_loops = real_predict[is_adjustment_measurements | is_loop].reset_index(drop=True)

        for index in range(1, len(adjustment_measurements) + 1):
            matches = adjustment_measurements_with_loops[adjustment_measurements_with_loops.to_index == index]
            self.algorithm.append(matches)

            if not self.draw_intermediate:
                continue

            if index % 100 == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt_df[:index]).to_global()

                noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements[:index]).to_global()
                self.evaluate(gt_trajectory, noised_trajectory,
                              f'test_real_predict_with_real_loops_only_noised_i_{index}')

                predicted_trajectory = self.algorithm.get_trajectory()
                self.evaluate(gt_trajectory, predicted_trajectory,
                              f'test_real_predict_with_real_loops_only_optimized_i_{index}')

                print(f'Saved. Index={index}')

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()

        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory,
                                      f'test_real_predict_with_real_loops_only_noised')

        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory,
                                         f'test_real_predict_with_real_loops_only_slam_optimized')

        print(' ')
        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)

    def test_real_predict_with_all_matches(self):
        # Data for testing is not ready
        return
        csv_path_gt = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_df = self.read_csv(csv_path_gt)

        csv_path_real_predict = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_mixed_real_predict.csv'
        real_predict = self.read_csv(csv_path_real_predict)

        is_adjustment_measurements = (real_predict.to_index - real_predict.from_index) <= 1
        adjustment_measurements = real_predict[is_adjustment_measurements].reset_index(drop=True)

        for index in range(len(adjustment_measurements)):
            matches = real_predict[real_predict.to_index == index]
            self.algorithm.append(matches)

            if not self.draw_intermediate:
                continue

            if index % 100 == 0:
                gt_trajectory = RelativeTrajectory.from_dataframe(gt_df[:index]).to_global()

                noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements[:index]).to_global()
                self.evaluate(gt_trajectory, noised_trajectory,
                              f'test_real_predict_with_all_matches_noised_i_{index}')

                predicted_trajectory = self.algorithm.get_trajectory()
                self.evaluate(gt_trajectory, predicted_trajectory,
                              f'test_gt_predict_with_all_matches_optimized_i_{index}')

                print(f'Saved. Index={index}')

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()

        noised_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()
        record_noised = self.evaluate(gt_trajectory, noised_trajectory,
                                      f'test_real_predict_with_real_loops_only_noised')

        predicted_trajectory = self.algorithm.get_trajectory()
        record_optimized = self.evaluate(gt_trajectory, predicted_trajectory,
                                         f'test_real_predict_with_real_loops_only_slam_optimized')

        print(' ')
        print('metrics before optimization', record_noised)
        print('metrics after optimization ', record_optimized)

        self.assert_greater(record_noised, record_optimized)


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
