import os
from functools import partial

import __init_path__
import env

from confidence_trainer import ConfidenceTrainer
from odometry.models import construct_flexible_model


class FlexibleWithConfidenceTrainer(ConfidenceTrainer):

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
        self.construct_model_fn = partial(construct_flexible_model,
                                          return_confidence=True)
        self.lr = 0.001
        self.loss = 'huber'
        self.scale_rotation = 50
        return super().get_model_factory(input_shapes)

    def get_callbacks(self, model, dataset, evaluate=True, save_dir=None):
        return super().get_callbacks(model=model,
                                     dataset=dataset,
                                     evaluate=evaluate,
                                     save_dir=save_dir)

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None):
        return super().fit_generator(model=model,
                                     dataset=dataset,
                                     epochs=epochs,
                                     evaluate=evaluate,
                                     save_dir=save_dir)

    def train(self):
        return super().train()


if __name__ == '__main__':
    parser = FlexibleWithConfidenceTrainer.get_parser()
    args = parser.parse_args()

    trainer = FlexibleWithConfidenceTrainer(**vars(args))
    trainer.train()
