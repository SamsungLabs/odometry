from tqdm import trange
import pandas as pd

from slam.utils import mlflow_logging
from slam.models.slam.base_slam import BaseSlam
from slam.models.relocalization import BoVW
from slam.aggregation import GraphOptimizer
from slam.keyframe_selector import CounterKeyFrameSelector


class RSlam(BaseSlam):

    @mlflow_logging(prefix='slam.', name='RSLAM')
    def __init__(self, keyframe_period, matches_threshold, matcher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold
        self.matcher = matcher

    def get_relocalization_model(self):
        reloc_model = BoVW(knn=self.knn, matches_threshold=self.matches_threshold, matcher=self.matcher)
        reloc_model.load(self.reloc_weights_path)
        return reloc_model

    def get_aggregator(self):
        return GraphOptimizer()

    def get_keyframe_selector(self):
        return CounterKeyFrameSelector(self.keyframe_period)

    def predict_generator(self, generator):

        self.init(generator[0][0][0][0])

        frame_history = pd.DataFrame()

        for frame_index in trange(1, len(generator)):
            x, y = generator[frame_index]
            image = x[0][0]
            prediction = self.predict(image)
            frame_history = frame_history.append(prediction, ignore_index=True)

        paths = generator.df.path_to_rgb.values
        frame_history['to_path'] = paths[frame_history.to_index.values.astype('int')]
        frame_history['from_path'] = paths[frame_history.from_index.values.astype('int')]

        return {'id': generator.trajectory_id,
                'frame_history': frame_history}

    def predict(self, frame):

        prediction = pd.DataFrame(columns=self.columns)
        prediction = prediction.append(self.get_matches(frame))

        self.last_frame = frame

        self.frame_index += 1

        return prediction
