import mlflow

import __init_path__
import env

from odometry.base_trainer import BaseTrainer
from image_retrieval.bovw import BoVW
import datetime
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
        self.voc_size = 128
        model = BoVW(self.voc_size)
        return model

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None):
        train_generator = dataset.get_train_generator()
        test_generator = dataset.get_test_generator()

        start_time = datetime.datetime.now()
        model.fit(train_generator)
        model.save(f'/home/d-zhukov/Projects/odometry/vocabulary_surf_{self.voc_size}.pkl')
        end_time = datetime.datetime.now()
        print(f' Calc time {(end_time - start_time).total_seconds()}')
        # matches = model.predict(test_generator)
        # return matches

    def train(self):

        dataset = self.get_dataset()

        model = self.get_model()

        matches = self.fit_generator(model=model, dataset=dataset, epochs=self.epochs)

        mlflow.log_param('successfully_finished', 1)
        mlflow.end_run()

        return matches


if __name__ == '__main__':

    parser = BoVWTrainer.get_parser()
    args = parser.parse_args()

    trainer = BoVWTrainer(**vars(args))
    matches = trainer.train()
