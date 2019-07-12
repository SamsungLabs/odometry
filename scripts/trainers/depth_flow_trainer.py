import os

import __init_path__
import env

from odometry.base_trainer import BaseTrainer
from odometry.models import construct_depth_flow_model


class DepthFlowTrainer(BaseTrainer):

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        self.x_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.image_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.load_mode = ['flow_xy', 'depth', 'depth']
        self.preprocess_mode = ['flow_xy', 'disparity', 'disparity']
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        self.construct_model_fn = construct_depth_flow_model
        self.lr = 0.0001
        self.loss = 'mae'
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

    parser = DepthFlowTrainer.get_parser()
    args = parser.parse_args()

    trainer = DepthFlowTrainer(**vars(args))
    trainer.train()
