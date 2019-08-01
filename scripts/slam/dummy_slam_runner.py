from slam.base_slam_runner import BaseSlamRunner
from slam.models import DummySlam


class DummySlamRunner(BaseSlamRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(knn=20, *args, **kwargs)

    def get_slam(self):
        return DummySlam(reloc_weights_path=self.reloc_weights,
                         optflow_weights_path=self.optflow_weights,
                         odometry_model_path=self.odometry_model,
                         knn=self.knn,
                         input_shapes=self.config['target_size'])


if __name__ == '__main__':

    parser = DummySlamRunner.get_parser()
    args = parser.parse_args()

    runner = DummySlamRunner(**vars(args))
    runner.run()
