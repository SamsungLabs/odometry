from slam.models.slam.base_slam import BaseSlam
from slam.models.relocalization import BoVW
from slam.keyframe_selector import CounterKeyFrameSelector
from slam.aggregation import DummyAverage
from slam.utils import mlflow_logging


class DummySlam(BaseSlam):

    @mlflow_logging(prefix='slam.', name='Dummy')
    def __init__(self, keyframe_period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyframe_period = keyframe_period

    def get_relocalization_model(self):
        reloc_model = BoVW(knn=self.knn)
        reloc_model.load(self.reloc_weights_path)
        return reloc_model

    def get_aggregator(self):
        return DummyAverage()

    def get_keyframe_selector(self):
        return CounterKeyFrameSelector(self.keyframe_period)
