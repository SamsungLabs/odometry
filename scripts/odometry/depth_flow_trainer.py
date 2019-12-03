from scripts.base_trainer import BaseTrainer
from slam.models import construct_depth_flow_model


class DepthFlowTrainer(BaseTrainer):

    def set_model_args(self):
        self.construct_model_fn = construct_depth_flow_model
        self.lr = 0.0001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.load_mode = ['flow_xy', 'depth', 'depth']
        self.preprocess_mode = ['flow_xy', 'disparity', 'disparity']


if __name__ == '__main__':

    parser = DepthFlowTrainer.get_parser()
    args = parser.parse_args()

    trainer = DepthFlowTrainer(**vars(args))
    trainer.train()
