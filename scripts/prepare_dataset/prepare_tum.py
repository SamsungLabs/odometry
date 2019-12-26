import os

import __init_path__
import env

from scripts.prepare_dataset.prepare_general import DatasetPreparator, get_default_dataset_parser


if __name__ == '__main__':
    width = 160
    height = 120
    parser = get_default_dataset_parser()
    parser.set_defaults(dataset_type='TUM',
                        dataset_root=os.path.join(env.DATASET_PATH, 'tum_rgbd/data/'),
                        target_size=(height, width),
                        matches_threshold=40,
                        relocalization_vocab_path=os.path.join(env.PROJECT_PATH, 'weights', 'tum_vocabulary.pkl'))
    args = parser.parse_args()
    DatasetPreparator(**vars(args)).prepare()
