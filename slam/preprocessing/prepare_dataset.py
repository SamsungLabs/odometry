import os
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

import env

from slam.utils.computation_utils import limit_resources
from slam.preprocessing import parsers, estimators, prepare_trajectory


def get_default_dataset_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--of_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/pwcnet.ckpt-84000'))
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--depth_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/model-199160'))
    parser.add_argument('--binocular_depth', action='store_true')
    parser.add_argument('--binocular_depth_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/pwcnet.ckpt-84000'))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--indices_root', type=str, default=None,
                        help='Path to directory with same structure as dataset. In trajectory subdirectories stored'
                             'df.csv files with new pair indices.')
    parser.add_argument('--matches_threshold', default=None, type=int)
    return parser


class DatasetPreparator:
    def __init__(self,
                 dataset_type,
                 dataset_root,
                 output_root,
                 target_size,
                 optical_flow_checkpoint,
                 depth_checkpoint=None,
                 binocular_depth_checkpoint=None,
                 pwc_features=False,
                 stride=1,
                 swap_angles=False,
                 indices_root=None,
                 matches_threshold=None):

        if self.indices_root:
            assert matches_threshold is not None

        self.dataset_type = dataset_type
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.target_size = target_size
        self.optical_flow_checkpoint = optical_flow_checkpoint
        self.depth_checkpoint = depth_checkpoint
        self.binocular_depth_checkpoint = binocular_depth_checkpoint
        self.pwc_features = pwc_features
        self.stride = stride
        self.swap_angles = swap_angles
        self.indices_root = indices_root
        self.matches_threshold = matches_threshold

    def _initialize_estimators(self):

        single_frame_estimators = []

        quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                          output_col=['euler_x', 'euler_y', 'euler_z'])
        single_frame_estimators.append(quaternion2euler_estimator)

        if self.depth_checkpoint is not None:
            struct2depth_estimator = estimators.Struct2DepthEstimator(input_col='path_to_rgb',
                                                                      output_col='path_to_depth',
                                                                      sub_dir='depth',
                                                                      checkpoint=self.depth_checkpoint,
                                                                      input_size=self.target_size)
            single_frame_estimators.append(struct2depth_estimator)

        if self.binocular_depth_checkpoint is not None:
            binocular_depth_estimator = estimators.BinocularDepthEstimator(
                input_col=['path_to_rgb', 'path_to_rgb_right',
                           'f_x', 'f_y', 'c_x', 'c_y', 'baseline_distance'],
                output_col='path_to_binocular_depth',
                sub_dir='binocular_depth',
                checkpoint=self.binocular_depth_checkpoint)
            single_frame_estimators.append(binocular_depth_estimator)

        cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        input_col = cols + [col + '_next' for col in cols]
        if self.swap_angles:
            output_col = ['euler_y', 'euler_x', 'euler_z', 't_x', 't_y', 't_z']
        else:
            output_col = cols
        global2relative_estimator = estimators.Global2RelativeEstimator(input_col=input_col,
                                                                        output_col=output_col)

        pwcnet_estimator = estimators.PWCNetEstimator(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                      output_col='path_to_optical_flow',
                                                      sub_dir='optical_flow',
                                                      checkpoint=self.optical_flow_checkpoint,
                                                      target_size=self.target_size)

        pair_frames_estimators = [global2relative_estimator, pwcnet_estimator]

        if self.pwc_features:
            features_extractor = estimators.PWCNetFeatureExtractor(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                                   output_col='path_to_features',
                                                                   sub_dir='features',
                                                                   checkpoint=self.optical_flow_checkpoint)
            pair_frames_estimators.append(features_extractor)

        return single_frame_estimators, pair_frames_estimators

    def _initialize_parser(self):
        # TODO: this is madness. We must get rid off getattr
        return getattr(parsers, f'{self.dataset_type}Parser')

    def _set_logger(self):
        fh = logging.FileHandler(self.output_root.joinpath('log.txt').as_posix(), mode='w+')
        fh.setLevel(logging.DEBUG)

        logger = logging.getLogger('prepare_dataset')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        return logger

    def prepare(self):
        limit_resources()

        if not isinstance(self.output_root, Path):
            self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        logger = self._set_logger()

        sf_estimators, pf_estimators = self._initialize_estimators()

        parser_class = self._initialize_parser()

        with open(self.output_root.joinpath('prepare_dataset.json').as_posix(), mode='w+') as f:
            dataset_config = {'depth_checkpoint': self.depth_checkpoint,
                              'optical_flow_checkpoint': self.optical_flow_checkpoint,
                              'target_size': self.target_size,
                              'stride': self.stride,
                              'binocular_depth_checkpoint': self.binocular_depth_checkpoint,
                              'matches_num': self.matches_threshold,
                              'indices_root': self.indices_root}
            json.dump(dataset_config, f)

        trajectories = [d.as_posix() for d in list(Path(self.dataset_root).rglob('*/**'))]
        trajectories.append(self.dataset_root)

        counter = 0

        for trajectory in tqdm(trajectories):
            try:
                trajectory_parser = parser_class(trajectory)
                trajectory_name = trajectory[len(self.dataset_root) + int(self.dataset_root[-1] != '/'):]
                output_dir = self.output_root.joinpath(trajectory_name)

                indices_path = os.path.join(self.indices_root, trajectory_name + '.csv') if self.indices_root else ''

                if not os.path.exists(indices_path):
                    logger.info(f'Indices file {indices_path} not found')

                logger.info(f'Preparing: {trajectory}. Output directory: {output_dir.as_posix()}.'
                            f'Indices path: {indices_path}')

                df = prepare_trajectory(output_dir,
                                        parser=trajectory_parser,
                                        single_frame_estimators=sf_estimators,
                                        pair_frames_estimators=pf_estimators,
                                        stride=self.stride,
                                        path_to_pair_indices=indices_path,
                                        matches_threshold=self.matches_threshold)
                df.to_csv(output_dir.joinpath('df.csv').as_posix(), index=False)

                counter += 1
                logger.info(f'Trajectory {trajectory} processed')

            except Exception as e:
                logger.info(e)

        logger.info(f'{counter} trajectories has been processed')
