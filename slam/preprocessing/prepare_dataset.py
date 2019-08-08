import os
import json
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from slam.utils.computation_utils import limit_resources
from slam.preprocessing import parsers, estimators, prepare_trajectory


def initialize_estimators(target_size,
                          optical_flow_checkpoint,
                          depth_checkpoint=None,
                          pwc_features=False,
                          swap_angles=False):

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
    if swap_angles:
        output_col = ['euler_y', 'euler_x', 'euler_z', 't_x', 't_y', 't_z']
    else:
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
    return getattr(parsers, f'{dataset_type}Parser')


def set_logger(output_dir):
    fh = logging.FileHandler(output_dir.joinpath('log.txt').as_posix(), mode='w+')
    fh.setLevel(logging.DEBUG)

    logger = logging.getLogger('prepare_dataset')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def prepare_dataset(dataset_type, dataset_root, output_root, target_size, optical_flow_checkpoint,
                    depth_checkpoint=None, pwc_features=False, stride=1, swap_angles=False):

    limit_resources()

    if not isinstance(output_root, Path):
        output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    set_logger(output_root)
    logger = logging.getLogger('prepare_dataset')

    sf_estimators, pf_estimators = initialize_estimators(target_size,
                                                         optical_flow_checkpoint=optical_flow_checkpoint,
                                                         depth_checkpoint=depth_checkpoint,
                                                         pwc_features=pwc_features,
                                                         swap_angles=swap_angles)

    parser_class = initialize_parser(dataset_type)

    with open(output_root.joinpath('prepare_dataset.json').as_posix(), mode='w+') as f:
        dataset_config = {'depth_checkpoint': depth_checkpoint,
                          'optical_flow_checkpoint': optical_flow_checkpoint,
                          'target_size': target_size,
                          'stride': stride}
        json.dump(dataset_config, f)

    trajectories = [d.as_posix() for d in list(Path(dataset_root).rglob('*/**'))]
    trajectories.append(dataset_root)

    counter = 0

    for trajectory in tqdm(trajectories):
        try:
            trajectory_parser = parser_class(trajectory)
            trajectory_name = trajectory[len(dataset_root) + int(dataset_root[:-1] != '/'):]
            output_dir = output_root.joinpath(trajectory_name)

            logger.info(f'Preparing: {trajectory}. Output directory: {output_dir.as_posix()}')

            df = prepare_trajectory(output_dir,
                                    parser=trajectory_parser,
                                    single_frame_estimators=sf_estimators,
                                    pair_frames_estimators=pf_estimators,
                                    stride=stride)
            df.to_csv(output_dir.joinpath('df.csv').as_posix(), index=False)

            counter += 1
            logger.info(f'Trajectory {trajectory} processed')

        except Exception as e:
            logger.info(e)

    logger.info(f'{counter} trajectories has been processed')
