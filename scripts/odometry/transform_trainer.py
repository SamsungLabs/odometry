from functools import partial

from scripts.base_trainer import BaseTrainer


class TransformTrainer(BaseTrainer):

    def __init__(self,
                 leader_board,
                 run_name,
                 bundle_name,
                 transform=None,
                 use_stride=False,
                 channel_wise=False,
                 concat_scale_to_fc=False,
                 multiply_outputs_by_scale=False,
                 **kwargs):

        super().__init__(leader_board=leader_board,
                         run_name=run_name,
                         bundle_name=bundle_name,
                         **kwargs)

        self.transform = transform
        self.use_stride = use_stride
        self.channel_wise = channel_wise
        self.concat_scale_to_fc = concat_scale_to_fc
        self.multiply_outputs_by_scale = multiply_outputs_by_scale
        self.placeholder = ['scale'] if self.multiply_outputs_by_scale else None

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        if self.use_stride:
            self.x_col.append('stride')
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        self.construct_model_fn = partial(self.construct_model_fn,
                                          transform=self.transform,
                                          channel_wise=self.channel_wise,
                                          agnostic=not self.use_stride,
                                          concat_scale_to_fc=self.concat_scale_to_fc,
                                          multiply_outputs_by_scale=self.multiply_outputs_by_scale)
        return super().get_model_factory(input_shapes)

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--transform', type=str, default=None,
                            choices=[None,
                                     'divide',
                                     'normalize',
                                     'project',
                                     'absmean_scale',
                                     'range_scale',
                                     'standard_scale',
                                     'percentile_scale'],
                            help='Inputs transform')
        parser.add_argument('--use_stride', action='store_true',
                            help='Use stride for scaling (if transform is not None)')
        parser.add_argument('--channel_wise', action='store_true',
                            help='Apply transform channel-wise (if transform is not None)')
        parser.add_argument('--concat_scale_to_fc', '-fc', action='store_true')
        parser.add_argument('--multiply_outputs_by_scale', '-mult', action='store_true')
        return parser
