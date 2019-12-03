from functools import partial

from scripts.base_trainer import BaseTrainer
from slam.models import construct_sequential_rt_model
from slam.linalg import Intrinsics


class SequentialRTTrainer(BaseTrainer):
    def __init__(self,
                 leader_board,
                 run_name,
                 bundle_name,
                 use_input_flow,
                 use_diff_flow,
                 use_rotation_flow,
                 f_x,
                 f_y,
                 c_x,
                 c_y,
                 **kwargs):

        super().__init__(leader_board=leader_board,
                         run_name=run_name,
                         bundle_name=bundle_name,
                         **kwargs)

        height, width = self.config['target_size']
        self.intrinsics = Intrinsics(f_x=f_x,
                                     f_y=f_y,
                                     c_x=c_x,
                                     c_y=c_y,
                                     width=width,
                                     height=height)

        self.use_input_flow = use_input_flow
        self.use_diff_flow = use_diff_flow
        self.use_rotation_flow = use_rotation_flow

    def set_model_args(self):
        self.construct_model_fn = construct_sequential_rt_model
        self.lr = 0.001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow']
        self.load_mode = ['flow_xy']
        self.preprocess_mode = ['flow_xy']

    def get_model_factory(self, input_shapes):
        self.construct_model_fn = partial(self.construct_model_fn,
                                          intrinsics=self.intrinsics,
                                          use_input_flow=self.use_input_flow,
                                          use_diff_flow=self.use_diff_flow,
                                          use_rotation_flow=self.use_rotation_flow)
        return super().get_model_factory(input_shapes)

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--use_input_flow', '-input', action='store_true')
        parser.add_argument('--use_diff_flow', '-diff', action='store_true')
        parser.add_argument('--use_rotation_flow', '-rotation', action='store_true')
        parser.add_argument('--f_x', type=float, default=0.5792554391619662)
        parser.add_argument('--f_y', type=float, default=1.91185106382978721)
        parser.add_argument('--c_x', type=float, default=0.48927703464947625)
        parser.add_argument('--c_y', type=float, default=0.4925949468085106)
        return parser


if __name__ == '__main__':

    parser = SequentialRTTrainer.get_parser()
    args = parser.parse_args()

    trainer = SequentialRTTrainer(**vars(args))
    trainer.train()
