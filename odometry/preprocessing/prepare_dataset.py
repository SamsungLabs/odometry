import os
import json
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

import __init_path__
import env

from odometry.utils.computation_utils import limit_resources
from odometry.preprocessing import parsers, estimators, prepare_trajectory


def initialize_estimators(target_size, optical_flow_checkpoint, depth_checkpoint=None, pwc_features=False):

    single_frame_estimators = list()

    quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                      output_col=['euler_x', 'euler_y', 'euler_z'])
    single_frame_estimators.append(quaternion2euler_estimator)

    if depth_checkpoint:
        struct2depth_estimator = estimators.Struct2DepthEstimator(input_col='path_to_rgb',
                                                                  output_col='path_to_depth',
                                                                  sub_dir='depth',
                                                                  checkpoint=depth_checkpoint,
                                                                  height=target_size[0],
                                                                  width=target_size[1])
        single_frame_estimators.append(struct2depth_estimator)

    cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
    input_col = cols + [col + '_next' for col in cols]
    output_col = cols
    global2relative_estimator = estimators.Global2RelativeEstimator(input_col=input_col,
                                                                    output_col=output_col)

    pwcnet_estimator = estimators.PWCNetEstimator(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                  output_col='path_to_optical_flow',
                                                  sub_dir='optical_flow',
                                                  checkpoint=optical_flow_checkpoint)

    pair_frames_estimators = [global2relative_estimator, pwcnet_estimator]

    if pwc_features:
        features_extractor = estimators.PWCNetFeatureExtractor(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                               output_col='path_to_features',
                                                               sub_dir='features',
                                                               checkpoint=optical_flow_checkpoint)
        pair_frames_estimators.append(features_extractor)

    return single_frame_estimators, pair_frames_estimators


def initialize_parser(dataset_type):
    if dataset_type == 'kitti':
        return parsers.KITTIParser
    elif dataset_type == 'discoman':
        return parsers.DISCOMANCSVParser
    elif dataset_type == 'tum':
        return parsers.TUMParser
    elif dataset_type == 'retailbot':
        return parsers.RetailBotParser
    else:
        raise RuntimeError('Unexpected dataset type')


def get_all_trajectories(dataset_root):

    if not isinstance(dataset_root, Path):
        dataset_root = Path(dataset_root)

    logger = logging.getLogger('prepare_dataset')

    trajectories = list()

    if list(dataset_root.glob('*traj.json')) or \
            list(dataset_root.glob('rgb.txt')) or \
            list(dataset_root.glob('image_2')) or \
            list(dataset_root.glob('camera_gt.csv')):

        logger.info(f'Trajectory {dataset_root.as_posix()} added')
        trajectories.append(dataset_root.as_posix())

    for d in dataset_root.rglob('**/*'):
        if list(d.glob('*traj.json')) or \
                list(d.glob('rgb.txt')) or \
                list(d.glob('image_2')) or \
                list(d.glob('camera_gt.csv')):

            logger.info(f'Trajectory {d.as_posix()} added')
            trajectories.append(d.as_posix())

    logger.info(f'Total: {len(trajectories)}')
    return trajectories


def set_logger(output_dir):
    fh = logging.FileHandler(output_dir.joinpath('log.txt').as_posix(), mode='w+')
    fh.setLevel(logging.DEBUG)

    logger = logging.getLogger('prepare_dataset')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def prepare_dataset(dataset_type, dataset_root, output_root, target_size, optical_flow_checkpoint,
                    depth_checkpoint=None, pwc_features=False):

    limit_resources()

    if not isinstance(output_root, Path):
        output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    set_logger(output_root)
    logger = logging.getLogger('prepare_dataset')

    sf_estimators, pf_estimators = initialize_estimators(target_size,
                                                         optical_flow_checkpoint=optical_flow_checkpoint,
                                                         depth_checkpoint=depth_checkpoint,
                                                         pwc_features=pwc_features)

    parser_class = initialize_parser(dataset_type)
    trajectories = get_all_trajectories(dataset_root)

    with open(output_root.joinpath('prepare_dataset.json').as_posix(), mode='w+') as f:
        dataset_config = {'depth_checkpoint': depth_checkpoint,
                          'optical_flow_checkpoint': optical_flow_checkpoint,
                          'target_size': target_size}
        json.dump(dataset_config, f)

    for trajectory in tqdm(trajectories):
        trajectory_name = trajectory[len(dataset_root) + int(dataset_root[-1] != '/'):]
        output_dir = output_root.joinpath(trajectory_name)

        try:
            trajectory_parser = parser_class(trajectory)

            df = prepare_trajectory(output_dir,
                                    parser=trajectory_parser,
                                    single_frame_estimators=sf_estimators,
                                    pair_frames_estimators=pf_estimators,
                                    stride=1)
            df.to_csv(output_dir.joinpath('df.csv').as_posix(), index=False)
        except Exception as e:
            logger.info(e)
            logger.info(f'WARNING! Trajectory {trajectory} failed to prepare')
            shutil.rmtree(output_dir.as_posix(), ignore_errors=True)
