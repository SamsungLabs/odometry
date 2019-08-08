import __init_path__
import env

from slam.base_slam_runner import BaseSlamRunner
from slam.models import DummySlam


class DummySlamRunner(BaseSlamRunner):

    def __init__(self, knn=20, keyframe_period=10, *args, **kwargs):
        super().__init__(knn=knn, *args, **kwargs)
        self.keyframe_period = keyframe_period

    def get_slam(self):
        return DummySlam(reloc_weights_path=self.reloc_weights,
                         optflow_weights_path=self.optflow_weights,
                         odometry_model_path=self.odometry_model,
                         knn=self.knn,
                         input_shapes=self.config['target_size'],
                         keyframe_period=self.keyframe_period)

    @staticmethod
    def get_parser():
        parser = BaseSlamRunner.get_parser()
        parser.add_argument('--keyframe_period', type=int, default=10, help='Period of keyframe selection')
        return parser


if __name__ == '__main__':

    parser = DummySlamRunner.get_parser()
    args = parser.parse_args()

    runner = DummySlamRunner(**vars(args))
    runner.run()
