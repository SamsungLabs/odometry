import unittest
import env
import os
from pathlib import Path
import pandas as pd
import shutil

from scripts.prepare_dataset import prepare_dataset

env.DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'minidataset')


class TestDatasets(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = Path(env.PROJECT_PATH).joinpath('tmp')
        shutil.rmtree(self.output_dir.as_posix(), ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir.as_posix())

    def assert_df(self, df, trajectory_dir, num_files):

        columns = ['path_to_rgb', 'path_to_depth', 'path_to_optical_flow']
        for column in columns:
            self.assertTrue(len(df[column]) == (num_files - 1), )
            for path in df[column]:
                file_path = os.path.join(trajectory_dir, path)
                self.assertTrue(os.path.isfile(file_path), f'Could not find file {file_path}')

        columns = ['path_to_rgb_next', 'path_to_depth_next']
        for column in columns:
            file_path = os.path.join(trajectory_dir, df[column].iloc[-1])
            self.assertTrue(os.path.isfile(file_path), f'Could not find file {file_path}')

    def test_tum(self) -> None:
        print('Started TUM test')

        num_files = 7
        prepare_dataset(dataset_type='tum',
                        dataset_root=os.path.join(env.DATASET_PATH, 'tum_rgbd_flow'),
                        output_root=self.output_dir.as_posix(),
                        target_size=(120, 160),
                        optical_flow_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/pwcnet.ckpt-595000'),
                        depth_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/model-199160')
                        )

        csv_path = list(self.output_dir.rglob("*.csv"))
        self.assertTrue(len(csv_path) == 1, f'Found {len(csv_path)} csv files')

        csv_path = csv_path[0]
        trajectory_dir = csv_path.parent
        df = pd.read_csv(csv_path)
        self.assert_df(df, trajectory_dir=trajectory_dir, num_files=num_files)

    def test_discoman(self) -> None:
        print('Started DISCOMAN test')

        num_files = 5

        prepare_dataset(dataset_type='discoman',
                        dataset_root=os.path.join(env.DATASET_PATH, 'renderbox'),
                        output_root=self.output_dir.as_posix(),
                        target_size=(120, 160),
                        optical_flow_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/pwcnet.ckpt-595000'),
                        depth_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/model-199160')
                        )

        csv_path = list(self.output_dir.rglob("*.csv"))
        self.assertTrue(len(csv_path) == 1, f'Found {len(csv_path)} csv files')

        csv_path = csv_path[0]
        trajectory_dir = csv_path.parent
        df = pd.read_csv(csv_path)
        self.assert_df(df, trajectory_dir=trajectory_dir, num_files=num_files)

    def test_kitti(self) -> None:
        print('Started KITTI test')

        num_files = 10

        prepare_dataset(dataset_type='kitti',
                        dataset_root=os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012'),
                        output_root=self.output_dir.as_posix(),
                        target_size=(120, 160),
                        optical_flow_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/pwcnet.ckpt-595000'),
                        depth_checkpoint=os.path.join(env.PROJECT_PATH, 'weights/model-199160')
                        )

        csv_path = list(self.output_dir.rglob("*.csv"))
        self.assertTrue(len(csv_path) == 1, f'Found {len(csv_path)} csv files')

        csv_path = csv_path[0]
        trajectory_dir = csv_path.parent
        df = pd.read_csv(csv_path)
        self.assert_df(df, trajectory_dir=trajectory_dir, num_files=num_files)
