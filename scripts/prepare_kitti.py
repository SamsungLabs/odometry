import argparse
import os

import env
<<<<<<< HEAD:scripts/prepare_kitti.py
from scripts import prepare_dataset
from odometry.utils import str2bool
=======
from scripts.prepare_dataset import prepare_dataset
from odometry.utils.utils import str2bool
>>>>>>> 70949e2... 1. Small file reorganization:scripts/prepare_kitti.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/'))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--of_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/pwcnet.ckpt-84000'))
    parser.add_argument('--depth', type=str2bool, default=True)
    parser.add_argument('--depth_checkpoint', type=str,
                        default=os.path.join(env.DATASET_PATH, 'Odometry_team/weights/model-199160'))
    args = parser.parse_args()

    prepare_dataset(dataset_type='kitti',
                    dataset_root=args.dataset_root,
                    output_root=args.output_dir,
                    target_size=(96, 320),
                    optical_flow_checkpoint=args.of_checkpoint,
                    depth_checkpoint=args.depth_checkpoint if args.depth else None)

