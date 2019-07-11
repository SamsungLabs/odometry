import os

import __init_path__
import env

from odometry.base_trainer import BaseTrainer
from odometry.models import construct_multiscale_model


class MultiscaleTrainer(BaseTrainer):

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        self.x_col = ['path_to_optical_flow']
        self.image_col = ['path_to_optical_flow']
        self.load_mode = ['flow_xy']
        self.preprocess_mode = ['flow_xy']
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        self.construct_model_fn = construct_multiscale_model
        self.lr = 0.001
        self.loss = 'huber'
        self.scale_rotation = 50
        return super().get_model_factory(input_shapes)

    def get_callbacks(self, model, dataset):
        return super().get_callbacks(model=model,
                                     dataset=dataset)

    def fit_generator(self, model, dataset, epochs):
        return super().fit_generator(model=model,
                                     dataset=dataset,
                                     epochs=epochs)

    def train(self):
        return super().train()


if __name__ == '__main__':

    parser = MultiscaleTrainer.get_parser()
    args = parser.parse_args()

    trainer = MultiscaleTrainer(**vars(args))
    trainer.train()
