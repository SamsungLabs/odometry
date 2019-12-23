import os
import mlflow

from scripts.base_trainer import BaseTrainer
from slam.models.relocalization import BoVW
from slam.data_manager.generator_factory import GeneratorFactory


class BoVWTrainer(BaseTrainer):

    def __init__(self, voc_size, train_sampling_step, matcher_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voc_size = voc_size
        self.train_sampling_step = train_sampling_step
        self.matcher_type = matcher_type

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):

        train_trajectories = train_trajectories or self.config['train_trajectories']
        val_trajectories = val_trajectories or self.config['val_trajectories']
        test_trajectories = self.config['test_trajectories']
        self.x_col = ['path_to_rgb']
        self.image_col = ['path_to_rgb']
        self.load_mode = 'rgb'
        self.preprocess_mode = 'rgb'

        self.run_dir = os.path.join(self.project_path, 'experiments', self.config['exp_name'], self.run_name)

        return GeneratorFactory(dataset_root=self.dataset_root,
                                train_trajectories=train_trajectories,
                                val_trajectories=val_trajectories,
                                test_trajectories=test_trajectories,
                                target_size=self.config['target_size'],
                                x_col=self.x_col,
                                image_col=self.image_col,
                                load_mode=self.load_mode,
                                preprocess_mode=self.preprocess_mode,
                                train_strides=self.train_sampling_step,
                                cached_images={})

    def get_model(self):
        model = BoVW(self.voc_size, run_dir=self.run_dir, matcher_type=self.matcher_type)
        return model

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None, prefix=None):
        train_generator = dataset.get_train_generator()
        model.fit(train_generator)

    def train(self):

        dataset = self.get_dataset()

        model = self.get_model()

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs)

        mlflow.log_param('successfully_finished', 1)
        mlflow.end_run()

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--voc_size', type=int, help='number of clusters to form vocabulary')
        parser.add_argument('--train_sampling_step', type=int)
        parser.add_argument('--matcher_type', type=str, choices=['BruteForce', 'Flann'])
        return parser


if __name__ == '__main__':

    parser = BoVWTrainer.get_parser()
    args = parser.parse_args()

    trainer = BoVWTrainer(**vars(args))
    trainer.train()
