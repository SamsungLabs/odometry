import argparse
import os

import __init_path__
import env

from slam.preprocessing import DatasetPreparator, get_default_dataset_parser
from slam.linalg import Intrinsics


if __name__ == '__main__':
    parser = get_default_dataset_parser()
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/'))
    args = parser.parse_args()
    width = 320
    height = 96
    intrinsics = Intrinsics(f_x=0.5792554391619662,
                            f_y=1.91185106382978721,
                            c_x=0.48927703464947625,
                            c_y=0.4925949468085106,
                            width=width,
                            height=height)
    baseline_distance = 0.54

    DatasetPreparator(dataset_type='KITTI',
                      dataset_root=args.dataset_root,
                      output_root=args.output_dir,
                      target_size=(height, width),
                      optical_flow_checkpoint=args.of_checkpoint,
                      stride=args.stride,
                      depth_checkpoint=args.depth_checkpoint if args.depth else None,
                      binocular_depth_checkpoint=args.binocular_depth_checkpoint if args.binocular_depth else None,
                      indices_root=args.indices_root,
                      intrinsics=intrinsics,
                      baseline_distance=baseline_distance).prepare()
