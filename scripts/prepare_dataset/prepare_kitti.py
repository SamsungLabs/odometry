import os

import __init_path__
import env

from scripts.prepare_dataset.prepare_general import DatasetPreparator, get_default_dataset_parser


if __name__ == '__main__':
    width = 320
    height = 96

    parser = get_default_dataset_parser()
    parser.set_defaults(dataset_type='KITTI',
                        dataset_root=os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/'),
                        target_size=(height, width),
                        matches_threshold=30,
                        relocalization_weights_path=os.path.join(env.PROJECT_PATH, 'weights', 'kitti_vocabulary.pkl'))
    args = parser.parse_args()
    DatasetPreparator(**vars(args)).prepare()
