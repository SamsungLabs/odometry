import argparse

import __init_path__

from scripts.graph_optimization import g2o_configs
from scripts.graph_optimization.base_search import BaseSearch
from slam.graph_optimization import TrajectoryEstimator


def main(dataset_root,
         config_type,
         strides,
         strides_sigmas,
         loop_sigma,
         loop_threshold,
         rotation_weight,
         max_iterations,
         vis_dir,
         pred_dir):

    assert len(strides) == len(strides_sigmas)
    strides_sigmas = {stride: weight for stride, weight in zip(strides, strides_sigmas)}
    config = getattr(g2o_configs, config_type)
    trajectory_names = BaseSearch().get_trajectory_names(config['1'][0])
    rpe_indices = BaseSearch().get_rpe_mode(config)
    X, y, groups = BaseSearch().get_data(config=config,
                                         dataset_root=dataset_root,
                                         trajectory_names=trajectory_names,
                                         val_mode='last')
    estimator = TrajectoryEstimator(strides_sigmas=strides_sigmas,
                                    loop_sigma=loop_sigma,
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
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--config_type', type=str, help='Name of config defined in g2o_configs.py')
    parser.add_argument('--strides', type=str, nargs='+',
                        help='List of strides. For these strides weights must be provided')
    parser.add_argument('--strides_sigmas', type=int, nargs='+',
                        help='Std of predictions from networks trained on different strides')
    parser.add_argument('--loop_sigma', type=int,
                        help='Std of prediction from network trained on relocalization estimator results')
    parser.add_argument('--loop_threshold', type=int,
                        help='Threshold value used to detect loops. Relocalization estimator returns pairs of similar '
                             'frames (frame_i, frame_j).'
                             'If (j - i) > loop_threshold then the pair of similar frames is considered to be the start'
                             'and the end of the loop')
    parser.add_argument('--rotation_weight', type=float,
                        help='weight of rotation constrains.')
    parser.add_argument('--max_iterations', type=int, default=5000,
                        help='Limit of iterations in g2o backend')
    parser.add_argument('--vis_dir', type=str, help='Path to visualization dir')
    parser.add_argument('--pred_dir', type=str, help='Path to prediction dir')

    args = parser.parse_args()
    main(**vars(args))
