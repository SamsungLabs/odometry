import __init_path__
import env

from slam.base_slam_runner import BaseSlamRunner
from slam.models import GraphSlam


class GraphSlamRunner(BaseSlamRunner):

    def __init__(self, knn=20,
                 keyframe_period=10,
                 matches_threshold=100,
                 max_iterations=1000,
                 online=False,
                 verbose=False,
                 matcher_type='Flann',
                 *args,
                 **kwargs):
        super().__init__(knn=knn, *args, **kwargs)
        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold
        self.max_iterations = max_iterations
        self.online = online
        self.verbose = verbose
        self.matcher_type = matcher_type

    def get_slam(self):
        return GraphSlam(reloc_weights_path=self.reloc_weights,
                         optflow_weights_path=self.optflow_weights,
                         odometry_model_path=self.odometry_model,
                         knn=self.knn,
                         optical_flow_shape=self.config['target_size'],
                         keyframe_period=self.keyframe_period,
                         matches_threshold=self.matches_threshold,
                         max_iterations=self.max_iterations,
                         online=self.online,
                         verbose=self.verbose,
                         matcher_type=self.matcher_type)

    @staticmethod
    def get_parser():
        p = BaseSlamRunner.get_parser()
        p.add_argument('--keyframe_period', type=int, default=10, help='Period of keyframe selection')
        p.add_argument('--matches_threshold', type=int, default=100, help='Parameter for BoVW')
        p.add_argument('--verbose', action='store_true')
        p.add_argument('--online', action='store_true', help='Optimize trajectory online')
        p.add_argument('--max_iterations', type=int, default=1000, help='Parameter for GraphOptimizer')
        p.add_argument('--matcher_type', type=str, choices=['BruteForce', 'Flann'])
        return p


if __name__ == '__main__':

    parser = GraphSlamRunner.get_parser()
    args = parser.parse_args()

    runner = GraphSlamRunner(**vars(args))
    runner.run()
