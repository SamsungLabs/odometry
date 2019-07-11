import os
import mlflow
from sklearn.model_selection import train_test_split

import __init_path__
import env

from odometry.base_trainer import BaseTrainer
from odometry.models import ModelWithConfidenceFactory


class ConfidenceTrainer(BaseTrainer):

    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 lsf=False,
                 batch=1,
                 epochs=100,
                 period=10,
                 save_best_only=False,
                 min_lr=1e-5,
                 reduce_factor=0.5,
                 holdout=0.1,
                 epochs_confidence=10):

        super().__init__(dataset_root=dataset_root,
                         dataset_type=dataset_type,
                         run_name=run_name,
                         lsf=lsf,
                         batch=batch,
                         epochs=epochs,
                         period=period,
                         save_best_only=save_best_only,
                         min_lr=min_lr,
                         reduce_factor=reduce_factor)
        self.holdout = holdout
        self.epochs_confidence = epochs_confidence

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None,
                    test_trajectories=None):
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories,
                                   test_trajectories=test_trajectories)

    def get_model_factory(self, input_shapes):
        return ModelWithConfidenceFactory(self.construct_model_fn,
                                          input_shapes=input_shapes,
                                          lr=self.lr,
                                          loss=self.loss,
                                          scale_rotation=self.scale_rotation)

    def get_callbacks(self, model, dataset):
        return super().get_callbacks(model=model,
                                     dataset=dataset)

    def fit_generator(self, model, dataset, epochs):
        return super().fit_generator(model=model,
                                     dataset=dataset,
                                     epochs=epochs)

    def train(self):
        train_trajectories = self.config['train_trajectories']
        train_trajectories, confidence_trajectories = train_test_split(train_trajectories,
                                                                       test_size=self.holdout,
                                                                       random_state=42)

        dataset = self.get_dataset(train_trajectories=train_trajectories)

        model_factory = self.get_model_factory(dataset.input_shapes)
        model = model_factory.construct()

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs)

        dataset_confidence = self.get_dataset(train_trajectories=confidence_trajectories)

        model = model_factory.freeze()

        self.fit_generator(model=model,
                           dataset=dataset_confidence,
                           epochs=self.epochs_confidence)

        mlflow.end_run()

    @staticmethod
    def get_parser():
        parser = super(ConfidenceTrainer, ConfidenceTrainer).get_parser()
        parser.add_argument('--holdout', type=float, default=0.1,
                            help='Ratio of dataset to train confidence')
        parser.add_argument('--epochs_confidence', type=int, default=10,
                            help='Number of epochs to train confidence')

        return parser
