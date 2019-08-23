from slam.models.slam.base_slam import BaseSlam
from slam.models.relocalization import BoVW
from slam.keyframe_selector import CounterKeyFrameSelector
from slam.aggregation import GraphOptimizer
from slam.utils import mlflow_logging


class GraphSlam(BaseSlam):

    @mlflow_logging(prefix='slam.', name='Dummy')
    def __init__(self, keyframe_period, matches_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold

    def get_relocalization_model(self):
        reloc_model = BoVW(knn=self.knn, matches_threshold=self.matches_threshold)
        reloc_model.load(self.reloc_weights_path)
        return reloc_model

    def get_aggregator(self):
        return GraphOptimizer(1000)

    def get_keyframe_selector(self):
        return CounterKeyFrameSelector(self.keyframe_period)
