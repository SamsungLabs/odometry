import argparse

import __init_path__

from scripts.graph_optimization import g2o_configs
from scripts.graph_optimization.base_search import Search
from slam.graph_optimization import TrajectoryEstimator


def main(config_type,
         strides,
         strides_std_weights,
         loop_std_weight,
         loop_threshold,
         rotation_weight,
         max_iterations,
         vis_dir,
         pred_dir):

    assert len(strides) == len(strides_std_weights)
    strides_std_weights = {stride: weight for stride, weight in zip(strides, strides_std_weights)}
    config = getattr(g2o_configs, config_type)
    trajectory_names = Search().get_trajectory_names(config['1'][0])
    rpe_indices = Search().get_rpe_mode(config)
    X, y, groups = Search().get_data(config,
                                     '/dbstore/datasets/Odometry_team/KITTI_odometry_2012_mixed/1/',
                                     trajectory_names,
                                     val_mode='last')
    estimator = TrajectoryEstimator(strides_std_weights=strides_std_weights,
                                    loop_std_weight=loop_std_weight,
                                    loop_threshold=loop_threshold,
                                    rotation_weight=rotation_weight,
                                    max_iterations=max_iterations,
                                    rpe_indices=rpe_indices,
                                    verbose=True,
                                    vis_dir=vis_dir,
                                    pred_dir=pred_dir)
    metrics = estimator.predict(X, y,  visualize=True, trajectory_names=trajectory_names)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_type', type=str, help='Name of config defined in g2o_configs.py')
    parser.add_argument('--strides', type=int, nargs='+',
                        help='List of strides. For these strides weights must be provided')
    parser.add_argument('--strides_std_weights', type=int, nargs='+',
                        help='Std weights of predictions from networks trained on different strides')
    parser.add_argument('--loop_std_weight', type=int,
                        help='Std weight of prediction from network trained on relocalization estimator results')
    parser.add_argument('--loop_threshold', type=int,
                        help='Relocalization estimator gives pairs of similar frames (frame_i, frame_j).'
                             'If (j - i) > loop_threshold than this pairs of similar images are considered as the start'
                             'and the end of the loop.')
    parser.add_argument('--rotation_weight', type=float,
                        help='weight of rotation constrains.')
    parser.add_argument('--max_iterations', type=int, default=5000,
                        help='Limit of iterations in g2o backend')
    parser.add_argument('--vis_dir', type=str, help='path to visualization dir')
    parser.add_argument('--pred_dir', type=str, help='path to prediction dir')

    args = parser.parse_args()
    main(**vars(args))
