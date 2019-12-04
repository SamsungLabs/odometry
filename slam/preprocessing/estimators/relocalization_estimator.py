import numpy as np
import os
import pandas as pd
from slam.models import BoVW
from slam.keyframe_selector import CounterKeyFrameSelector

from .network_estimator import NetworkEstimator
from slam.utils import load_image, resize_image_arr


class RelocalizationEstimator(NetworkEstimator):

    def __init__(self,
                 keyframe_period,
                 matches_threshold,
                 knn,
                 weights_path,
                 target_size,
                 matcher_type='BruteForce',
                 *args,
                 **kwargs):

        super(NetworkEstimator, self).__init__(output_col=None, *args, **kwargs)
        self.name = 'RelocalizationEstimator'
        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold
        self.matcher_type = matcher_type
        self.knn = knn
        self.weights_path = weights_path
        self.target_size = target_size
        self.relocalization_columns = ['to_index', 'from_index', 'to_db_index', 'from_db_index']

        self.reloc_model = None
        self.keyframe_selector = None

        self.last_frame = None
        self.last_keyframe = None

        self.frame_index = 0

    def init(self, image):
        self.reloc_model = self._load_model()
        self.keyframe_selector = self._get_keyframe_selector()

        self.last_frame = image
        self.last_keyframe = image

        self.reloc_model.add(self.last_keyframe, 0)
        self.frame_index = 1

    def _load_model(self):
        reloc_model = BoVW(knn=self.knn, matches_threshold=self.matches_threshold, matcher_type=self.matcher_type)
        reloc_model.load(self.weights_path)
        return reloc_model

    def _load_model_input(self, row, dataset_root):
        image = load_image(os.path.join(dataset_root, row[self.input_col]))
        image = resize_image_arr(image,
                                 self.target_size,
                                 data_format='channels_last',
                                 mode='nearest')
        return image

    def _run_model_inference(self, model_input):

        if self.frame_index == 0:
            self.init(model_input)

        matches = list()
        match = pd.Series({'to_index': self.frame_index, 'from_index': self.frame_index - 1})
        matches.append(match)

        new_key_frame = self.keyframe_selector.is_key_frame(self.last_keyframe, model_input, self.frame_index)
        if new_key_frame:
            predicts = self.reloc_model.predict(model_input, self.frame_index)
            for index, predict in predicts.iterrows():
                matches = matches.append(predict)
            self.last_keyframe = model_input

        self.last_frame = model_input
        self.frame_index += 1
        return matches

    def _get_keyframe_selector(self):
        return CounterKeyFrameSelector(self.keyframe_period)

    def run(self, row: pd.Series, dataset_root: str):
        model_input = self._load_model_input(row, dataset_root)
        prediction = self.predict(model_input)
        rows = [row.append(predicted_row) for predicted_row in prediction]
        return rows
