import os

import __init_path__
import env

from scripts.prepare_dataset.prepare_general import DatasetPreparator, get_default_dataset_parser


if __name__ == '__main__':
    width = 160
    height = 120
    parser = get_default_dataset_parser()
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join(env.DATASET_PATH, 'zju')),
    args = parser.parse_args()


    DatasetPreparator(dataset_type='ZJU',
                      target_size=(height, width),
                      relocalization_weights_path=os.path.join(env.PROJECT_PATH, 'weights', 'euroc_vocabulary.pkl'),
                      **vars(args)).prepare()
