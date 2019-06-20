import os
import __init_path__
import env
import mlflow
from odometry.utils import make_memory_safe
from odometry.preprocessing import parsers, estimators, prepare_trajectory
from odometry.data_manager import GeneratorFactory
import itertools


def initialize_parser():
    trajectory_dir = 'discoman'
    json_path = os.path.join(env.DATASET_PATH, '/renderbox/iros2019/dset/output/deprecated/000001/0_traj.json')
    parser = parsers.DISCOMANParser(trajectory_dir, json_path)
    return parser


def initialize_estimators(target_size):
    quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                      output_col=['euler_x', 'euler_y', 'euler_z'])

    depth_checkpoint = os.path.abspath('../weights/model-199160')
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

    optical_flow_checkpoint = '/Vol0/user/f.konokhov/tfoptflow/tfoptflow/tmp/pwcnet.ckpt-84000'
    # optical_flow_checkpoint = os.path.abspath(../weights/pwcnet.ckpt-595000')  # official weights
    pwcnet_estimator = estimators.PWCNetEstimator(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                  output_col='path_to_optical_flow',
                                                  sub_dir='optical_flow',
                                                  checkpoint=optical_flow_checkpoint)

    single_frame_estimators = [quaternion2euler_estimator, struct2depth_estimator]
    pair_frames_estimators = [global2relative_estimator, pwcnet_estimator]
    return single_frame_estimators, pair_frames_estimators


def prepare_dataset(trajectory_names, target_size):
    parser = initialize_parser()
    single_frame_estimators, pair_frames_estimators = initialize_estimators(target_size)
    for trajectory_name in trajectory_names:
        trajectory_dir = 'discoman/{}'.format(trajectory_name)
        df = prepare_trajectory(trajectory_dir,
                                parser=parser,
                                single_frame_estimators=single_frame_estimators,
                                pair_frames_estimators=pair_frames_estimators,
                                stride=1)
        df.to_csv(os.path.join(trajectory_dir, 'df.csv'), index=False)

config = {
    "train_sequences": [
        "train/0085",
    ],
    "val_sequences": [
        "val/0007"
    ]
}

target_size = (120, 160)

make_memory_safe(prepare_dataset, cuda_visible_devices=CUDA_VISIBLE_DEVICES)(
    itertools.chain(config['train_sequences'], config['val_sequences']),
    target_size=target_size