import os
import mlflow
from sklearn.model_selection import train_test_split

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import ModelWithConfidenceFactory
from slam.data_manager import GeneratorFactory


class ConfidenceTrainer(BaseTrainer):

    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 seed=42,
                 lsf=False,
                 cache=False,
                 batch=1,
                 epochs=100,
                 period=10,
                 save_best_only=False,
                 min_lr=1e-5,
                 reduce_factor=0.5,
                 holdout=0.1,
                 epochs_confidence=100):

        super().__init__(dataset_root=dataset_root,
                         dataset_type=dataset_type,
                         run_name=run_name,
                         seed=seed,
                         lsf=lsf,
                         cache=cache,
                         batch=batch,
                         epochs=epochs,
                         period=period,
                         save_best_only=save_best_only,
                         min_lr=min_lr,
                         reduce_factor=reduce_factor)
        self.holdout = holdout
        self.epochs_confidence = epochs_confidence
        self.max_to_visualize = 0
        self.placeholder = ['confidence']

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        return ModelWithConfidenceFactory(self.construct_model_fn,
                                          input_shapes=input_shapes,
                                          lr=self.lr,
                                          loss=self.loss,
                                          scale_rotation=self.scale_rotation)

    def get_callbacks(self, model, dataset, evaluate=True, save_dir=None, prefix=None):
        return super().get_callbacks(model=model,
                                     dataset=dataset,
                                     evaluate=evaluate,
                                     save_dir=save_dir,
                                     prefix=prefix)

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None, prefix=None):
        return super().fit_generator(model=model,
                                     dataset=dataset,
                                     epochs=epochs,
                                     evaluate=evaluate,
                                     save_dir=save_dir,
                                     prefix=prefix)

    def train(self):
        dataset = self.get_dataset()

        train_index, confidence_index = train_test_split(dataset.df_train.index,
                                                         test_size=self.holdout,
                                                         random_state=42)
        df_train = dataset.df_train.loc[train_index]
        df_confidence = dataset.df_train.loc[confidence_index]

        dataset.df_train = df_train

        model_factory = self.get_model_factory(dataset.input_shapes)
        model = model_factory.construct()

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs,
                           evaluate=False,
                           save_dir='dof')

        dataset.df_train = df_confidence

        model = model_factory.freeze()
        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs_confidence,
                           evaluate=False,
                           save_dir='confidence',
                           prefix='confidence')

        mlflow.log_metric('successfully_finished', 1)
        mlflow.end_run()

    @staticmethod
    def get_parser():
        parser = super().get_parser()
        parser.add_argument('--holdout', type=float, default=0.1,
                            help='Ratio of dataset to train confidence')
        parser.add_argument('--epochs_confidence', type=int, default=100,
                            help='Number of epochs to train confidence')

        return parser
