import os
import pandas as pd
from slam.models import BoVW

from slam.utils import load_image_arr
from .network_estimator import NetworkEstimator
from slam.keyframe_selector import CounterKeyFrameSelector


class RelocalizationEstimator(NetworkEstimator):

    def __init__(self,
                 keyframe_period,
                 matches_threshold,
                 knn,
                 target_size,
                 matcher_type='BruteForce',
                 *args,
                 **kwargs):

        self.keyframe_period = keyframe_period
        self.matches_threshold = matches_threshold
        self.matcher_type = matcher_type
        self.knn = knn
        self.target_size = target_size
        self.relocalization_columns = ['to_index', 'from_index', 'to_db_index', 'from_db_index']

        self.reloc_model = None
        self.keyframe_selector = None

        self.last_frame = None
        self.last_keyframe = None

        self.frame_index = 0

        super(RelocalizationEstimator, self).__init__(ext='csv',
                                                      name='Relocalization',
                                                      *args,
                                                      **kwargs)

    def _load_model(self):
        self.reloc_model = BoVW(knn=self.knn,
                                matches_threshold=self.matches_threshold,
                                matcher_type=self.matcher_type)
        self.reloc_model.load(self.checkpoint)
        self.keyframe_selector = self._get_keyframe_selector()

    def _load_model_input(self, row, dataset_root):
        image = load_image_arr(os.path.join(dataset_root, row[self.input_col]), mode='RGB')
        return image

    def _run_model_inference(self, model_input):
        if self.frame_index == 0:
            self.last_frame = model_input
            self.last_keyframe = model_input

            self.reloc_model.add(self.last_keyframe, 0)
            self.frame_index = 1
            return pd.DataFrame({'to_index': [], 'from_index': []})

        matches = pd.DataFrame({'to_index': [self.frame_index], 'from_index': [self.frame_index - 1]})
        new_key_frame = self.keyframe_selector.is_key_frame(self.last_keyframe, model_input, self.frame_index)
        if new_key_frame:
            predicts = self.reloc_model.predict(model_input, self.frame_index)
            matches = matches.append(predicts).reset_index(drop=True)
            self.last_keyframe = model_input

        self.last_frame = model_input
        self.frame_index += 1
        return matches

    def _get_keyframe_selector(self):
        return CounterKeyFrameSelector(self.keyframe_period)

    def _save_model_prediction(self, model_output, row, dataset_root):
        os.makedirs(os.path.join(dataset_root, self.dir), exist_ok=True)
        output_path = os.path.join(self.dir, self._create_output_filename(row))
        model_output.to_csv(os.path.join(dataset_root, output_path))
        return output_path

    def run(self, row: pd.Series, dataset_root: str):
        model_input = self._load_model_input(row, dataset_root)
        model_output = self.predict(model_input)
        self._save_model_prediction(model_output, row, dataset_root)
        row[self.output_col] = model_output.from_index.values
        return row
