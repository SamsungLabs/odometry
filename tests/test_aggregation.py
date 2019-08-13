import os
import unittest
import numpy as np
import pandas as pd

import __init_path__
import env

from slam.evaluation import calculate_metrics
from slam.linalg import RelativeTrajectory


class BaseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.algorithm = None

    def gt2predict(self, gt, stride):
        columns_mean = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        predict = gt[columns_mean]
        columns_std = [c + '_confidence' for c in columns_mean]

        predict[columns_std] = pd.DataFrame([[1] * len(columns_std)] * len(gt), index=predict.index)
        predict['to_index'] = np.arange(0, len(gt), stride) + 1
        predict['from_index'] = np.arange(0, len(gt), stride)
        return predict

    def test_1(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt, stride=1)
        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')

        self.assertAlmostEqual(record['ATE'], 0)
        self.assertAlmostEqual(record['RPE'], 0)
        self.assertAlmostEqual(record['RMSE_t'], 0)
        self.assertAlmostEqual(record['RMSE_r'], 0)

    def test_2(self):
        csv_path_s1 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_1.csv'
        gt_s1 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s1))
        pred_s1 = self.gt2predict(gt_s1, stride=1)
        self.algorithm.append(pred_s1)

        csv_path_s2 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/03_stride_2.csv'
        gt_s2 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s2))
        pred_s2 = self.gt2predict(gt_s2, stride=2)
        self.algorithm.append(pred_s2)

        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_s1).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')

        self.assertAlmostEqual(record['ATE'], 0)
        self.assertAlmostEqual(record['RPE'], 0)
        self.assertAlmostEqual(record['RMSE_t'], 0)
        self.assertAlmostEqual(record['RMSE_r'], 0)

    def test_3(self):
        csv_path_s1 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_1.csv'
        gt_s1 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s1))
        pred_s1 = self.gt2predict(gt_s1, stride=1)
        self.algorithm.append(pred_s1)

        csv_path_s2 = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_2.csv'
        gt_s2 = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path_s2))
        pred_s2 = self.gt2predict(gt_s2, stride=2)
        self.algorithm.append(pred_s2)

        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt_s1).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')

        self.assertAlmostEqual(record['ATE'], 0)
        self.assertAlmostEqual(record['RPE'], 0)
        self.assertAlmostEqual(record['RMSE_t'], 0)
        self.assertAlmostEqual(record['RMSE_r'], 0)

    def test_4(self):
        csv_path = 'tests/minidataset/KITTI_odometry_2012/dataset/dataframes/00_stride_mixed.csv'
        gt = pd.read_csv(os.path.join(env.PROJECT_PATH, csv_path))
        pred = self.gt2predict(gt, stride=1)
        self.algorithm.append(pred)
        predicted_trajectory = self.algorithm.get_trajectory()

        gt_trajectory = RelativeTrajectory.from_dataframe(gt).to_global()
        record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full')

        self.assertAlmostEqual(record['ATE'], 0)
        self.assertAlmostEqual(record['RPE'], 0)
        self.assertAlmostEqual(record['RMSE_t'], 0)
        self.assertAlmostEqual(record['RMSE_r'], 0)

