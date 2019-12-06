import os
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

import env

from slam.utils.computation_utils import limit_resources
from slam.preprocessing import parsers, estimators, prepare_trajectory


def get_default_dataset_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, help='Base name of parser.')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--target_size', type=int, nargs='+', help='Size of images')
    parser.add_argument('--optical_flow_checkpoint', '--of_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/pwcnet.ckpt-84000'))
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--depth_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/model-199160'))
    parser.add_argument('--binocular_depth', action='store_true')
    parser.add_argument('--binocular_depth_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/pwcnet.ckpt-84000'))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--trajectories', default=None, type=str, nargs='+', help='Name of trajectories')
    parser.add_argument('--relocalization', action='store_true')
    parser.add_argument('--max_matches', default=20, type=str,
                        help='Maximum number of matches that relocalization'
                             'estimator can find')
    parser.add_argument('--keyframe_period', default=2, type=int,
                        help='Keyframe are considered as every N-th frame,'
                             'where N is keyframe period.')
    parser.add_argument('--matches_threshold', default=40, type=int,
                        help='This threshold of matched keypoints number'
                             'needed for additional check how good two'
                             'images are matched.'
                             'General rule of thumb:'
                             '    for KITTI 30,'
                             '    for TUM 40, '
                             '    for EuRoC 40')
    parser.add_argument('--relocalization_weights_path', type=str,
                        help='Weights for relocalization model')
    return parser


class DatasetPreparator:
    def __init__(self,
                 dataset_type,
                 dataset_root,
                 output_root,
                 target_size,
                 undistort=False,
                 relocalization=False,
                 optical_flow_checkpoint=None,
                 depth=False,
                 depth_checkpoint=None,
                 binocular_depth=None,
                 binocular_depth_checkpoint=None,
                 pwc_features=False,
                 stride=1,
                 swap_angles=False,
                 trajectories=None,
                 max_matches=None,
                 keyframe_period=None,
                 matches_threshold=None,
                 relocalization_weights_path=None):

        self.dataset_type = dataset_type
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.target_size = target_size
        self.undistort = undistort
        self.relocalization = relocalization
        self.optical_flow_checkpoint = optical_flow_checkpoint
        self.depth_checkpoint = depth_checkpoint if depth else None
        self.binocular_depth_checkpoint = binocular_depth_checkpoint if binocular_depth else None
        self.pwc_features = pwc_features
        self.stride = stride
        self.swap_angles = swap_angles
        self.trajectories = trajectories
        self.matches_threshold = matches_threshold
        self.max_matches = max_matches
        self.keyframe_period = keyframe_period
        self.relocalization_weights_path = relocalization_weights_path

    def _initialize_estimators(self):

        single_frame_estimators = []

        quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                          output_col=['euler_x', 'euler_y', 'euler_z'])
        single_frame_estimators.append(quaternion2euler_estimator)

        if self.undistort:
            undistortion_estimator = estimators.UndistortionEstimator(
                input_col=['path_to_rgb', 'K', 'D', 'R', 'P'],
                output_col='path_to_rgb',
                sub_dir='rgb_undistorted')
            single_frame_estimators.append(undistortion_estimator)

            undistortion_estimator_right = estimators.UndistortionEstimator(
                input_col=['path_to_rgb_right', 'K_right', 'D_right', 'R_right', 'P'],
                output_col='path_to_rgb_right',
                sub_dir='rgb_undistorted_right')
            single_frame_estimators.append(undistortion_estimator_right)

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

        if self.relocalization:
            relocalization_estimator = estimators.RelocalizationEstimator(input_col='path_to_rgb',
                                                                          output_col='from_index',
                                                                          sub_dir='reloc',
                                                                          knn=self.max_matches,
                                                                          matches_threshold=self.matches_threshold,
                                                                          keyframe_period=self.keyframe_period,
                                                                          checkpoint=self.relocalization_weights_path,
                                                                          target_size=self.target_size)
            single_frame_estimators.append(relocalization_estimator)

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
                              'matches_num': self.matches_threshold}
            json.dump(dataset_config, f)

        if self.trajectories is None:
            trajectories = [d.as_posix() for d in list(Path(self.dataset_root).rglob('*/**'))]
            trajectories.append(self.dataset_root)
        else:
            trajectories = [os.path.join(self.dataset_root, trajectory) for trajectory in self.trajectories]

        counter = 0

        for trajectory in tqdm(trajectories):
            try:
                trajectory_parser = parser_class(trajectory)
                trajectory_name = trajectory[len(self.dataset_root) + int(self.dataset_root[-1] != '/'):]
                output_dir = self.output_root.joinpath(trajectory_name)

                logger.info(f'Preparing: {trajectory}. Output directory: {output_dir.as_posix()}.')

                df = prepare_trajectory(output_dir,
                                        parser=trajectory_parser,
                                        single_frame_estimators=sf_estimators,
                                        pair_frames_estimators=pf_estimators,
                                        stride=self.stride)
                df.to_csv(output_dir.joinpath('df.csv').as_posix(), index=False)

                counter += 1
                logger.info(f'Trajectory {trajectory} processed')

            except Exception as e:
                logger.info(e)

        logger.info(f'{counter} trajectories has been processed')
