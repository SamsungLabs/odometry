import mlflow

import os
import __init_path__
import env

from odometry.base_trainer import BaseTrainer
from relocalization.bovw import BoVW
from odometry.data_manager.generator_factory import GeneratorFactory


class BoVWTrainer(BaseTrainer):

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
                                train_sampling_step=50,
                                cached_images={})

    def get_model(self):
        self.voc_size = 512
        model = BoVW(self.voc_size)
        return model

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None):
        train_generator = dataset.get_train_generator()

        model.fit(train_generator)

        if mlflow.active_run:
            model.save(os.path.join(self.run_dir, f'vocabulary.pkl'))
            mlflow.log_artifact(self.run_dir, self.run_name)

    def train(self):

        dataset = self.get_dataset()

        model = self.get_model()

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs)

        mlflow.log_param('successfully_finished', 1)
        mlflow.end_run()


if __name__ == '__main__':

    parser = BoVWTrainer.get_parser()
    args = parser.parse_args()

    trainer = BoVWTrainer(**vars(args))
    trainer.train()
