import os

import __init_path__
import env

from scripts.prepare_dataset.prepare_general import DatasetPreparator, get_default_dataset_parser


if __name__ == '__main__':
    width = 160
    height = 90

    parser = get_default_dataset_parser()
    parser.set_defaults(dataset_type='DISCOMAN',
                        dataset_root=os.path.join(env.DATASET_PATH, 'Odometry_team/discoman_v10_unzip/'),
                        target_size=(height, width),
                        swap_angles=True,
                        matches_threshold=3,
                        # This number is random should be refined with relocalization_visualization.ipnb
                        relocalization_vocab_path=os.path.join(env.PROJECT_PATH, 'weights', 'kitti_vocabulary.pkl'))
    args = parser.parse_args()

    DatasetPreparator(**vars(args)).prepare()
