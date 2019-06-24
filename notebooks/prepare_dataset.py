import os
import argparse
import __init_path__
import env
import json
from odometry.preprocessing import parsers, estimators, prepare_trajectory
import itertools
import odometry.preprocessing.splits as configs
from tqdm import tqdm

def initialize_estimators(target_size, optical_flow_checkpoint, depth_checkpoint):
    quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                      output_col=['euler_x', 'euler_y', 'euler_z'])

    struct2depth_estimator = estimators.Struct2DepthEstimator(input_col='path_to_rgb',
                                                              output_col='path_to_depth',
                                                              sub_dir='depth',
                                                              checkpoint=depth_checkpoint,
                                                              height=target_size[0],
                                                              width=target_size[1])

    cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
    input_col = cols + [col + '_next' for col in cols]
    output_col = cols
    global2relative_estimator = estimators.Global2RelativeEstimator(input_col=input_col,
                                                                    output_col=output_col)

    pwcnet_estimator = estimators.PWCNetEstimator(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                  output_col='path_to_optical_flow',
                                                  sub_dir='optical_flow',
                                                  checkpoint=optical_flow_checkpoint)

    single_frame_estimators = [quaternion2euler_estimator, struct2depth_estimator]
    pair_frames_estimators = [global2relative_estimator, pwcnet_estimator]
    return single_frame_estimators, pair_frames_estimators


def initialize_parser(dataset_type, trajectory_id):
    parser = None
    if dataset_type == "kitti":
        parser = parsers.KITTIParser(trajectory_dir='', trajectory_id=trajectory_id)

    return parser

def initialize(datset_type):

    if args.dataset == "kitti":
        config = configs.get_kitti_config_1()
        parser = parsers.KITTIParser
    elif args.dataset == "discoman":
        config = configs.get_discoman_iros_1_config()
        parser = parsers.DISCOMANParser

    elif args.dataset == "tum":
        config = configs.get_tum_config()
        parser = parsers.TUMParser
    else:
        raise RuntimeError("Unexpected dataset type")

    trajectories = itertools.chain(config['train_sequences'], config['val_sequences'], config['test_sequences'])
    return parser, trajectories


def prepare_dataset(dataset_type, dataset_root, output_dir, target_size, optical_flow_checkpoint, depth_checkpoint):

    single_frame_estimators, pair_frames_estimators = initialize_estimators(target_size,
                                                                            optical_flow_checkpoint=optical_flow_checkpoint,
                                                                            depth_checkpoint=depth_checkpoint)

    parser, trajectories = initialize(dataset_type)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, "config.json"), mode="w+") as f:
        dataset_config = {"depth_checkpoint": depth_checkpoint, "optical_flow_checkpoint": optical_flow_checkpoint}
        json.dump(dataset_config, f)

    for trajectory in tqdm(trajectories):
        parser = parser(trajectory, dataset_root)
        trajectory_dir = os.path.join(output_dir, trajectory)
        df = prepare_trajectory(trajectory_dir,
                                parser=parser,
                                single_frame_estimators=single_frame_estimators,
                                pair_frames_estimators=pair_frames_estimators,
                                stride=1)
        df.to_csv(os.path.join(trajectory_dir, 'df.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="possible variants: kitti, discoman, tum")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--of_checkpoint", type=str, default='/Vol0/user/f.konokhov/tfoptflow/tfoptflow/tmp/pwcnet.ckpt-84000')
    parser.add_argument("--depth_checkpoint", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                                             'weights/model-199160'))
    args = parser.parse_args()

    target_size = (120, 160)

    prepare_dataset(args.dataset, dataset_root=args.dataset_root, output_dir=args.output_dir, target_size=target_size,
                    optical_flow_checkpoint=args.of_checkpoint, depth_checkpoint=args.depth_checkpoint)


