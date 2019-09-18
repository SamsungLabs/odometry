import __init_path__
import env

from slam.base_slam_runner import BaseSlamRunner
from slam.models import RSlam


class RSlamRunner(BaseSlamRunner):

    def __init__(self, knn=20,
                 keyframe_period=10,
                 matches_threshold=100,
                 max_iterations=1000,
                 online=False,
                 verbose=False,
                 matcher='Flann',
                 *args,
                 **kwargs):
        super().__init__(knn=knn, *args, **kwargs)
        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold
        self.max_iterations = max_iterations
        self.online = online
        self.verbose = verbose
        self.matcher = matcher

    def get_slam(self):
        return RSlam(reloc_weights_path=self.reloc_weights,
                     optflow_weights_path=self.optflow_weights,
                     odometry_model_path=self.odometry_model,
                     knn=self.knn,
                     optical_flow_shape=self.config['target_size'],
                     keyframe_period=self.keyframe_period,
                     matches_threshold=self.matches_threshold,
                     matcher=self.matcher)

    @staticmethod
    def get_parser():
        p = BaseSlamRunner.get_parser()
        p.add_argument('--keyframe_period', type=int, default=10, help='Period of keyframe selection')
        p.add_argument('--matches_threshold', type=int, default=100, help='Parameter for BoVW')
        p.add_argument('--matcher', type=str, choices=['BruteForce', 'Flann'])
        return p


if __name__ == '__main__':

    parser = RSlamRunner.get_parser()
    args = parser.parse_args()

    runner = RSlamRunner(**vars(args))
    runner.run()
