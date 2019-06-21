import os
import numpy as np

from odometry.utils import load_image
from odometry.preprocessing.estimators.base_estimator import BaseEstimator


class NetworkEstimator(BaseEstimator):

    def __init__(self,
                 input_col,
                 output_col,
                 checkpoint,
                 sub_dir,
                 height=None,
                 width=None):
        super(NetworkEstimator, self).__init__(input_col, output_col)

        self.height = height
        self.width = width

        self.dir = sub_dir

        self.checkpoint = checkpoint
        self._load_model()

        self.name = 'Network'

    def _load_model(self):
        raise NotImplementedError

    def _convert_image_to_model_input(self, image):
        return np.array(image, dtype=np.float32)

    def _convert_model_output_to_prediction(self, output):
        return output

    def _create_output_filename(self, row):
        input_col_as_list = [self.input_col] if isinstance(self.input_col, str) else self.input_col
        input_basenames = [os.path.basename(filepath) for filepath in row[input_col_as_list]]
        output_filename = '_'.join([os.path.splitext(basename)[0] for basename in input_basenames])
        output_filename = '.'.join((output_filename, self.ext))
        return output_filename

    def _load_model_input(self, row, dataset_root):
        if isinstance(self.input_col, str):
            model_input = self._convert_image_to_model_input(
                load_image(os.path.join(dataset_root, row[self.input_col])))[None]
        else:
            model_input = [[self._convert_image_to_model_input(
                    load_image(os.path.join(dataset_root, row[input_col])))
                for input_col in self.input_col]]
        return model_input

    def _save_model_output(self, model_output, row, dataset_root):
        os.makedirs(os.path.join(dataset_root, self.dir), exist_ok=True)
        output_path = os.path.join(self.dir, self._create_output_filename(row))
        np.save(os.path.join(dataset_root, output_path), self._convert_model_output_to_prediction(model_output))
        return output_path

    def _run_model_inference(self, model_input):
        raise NotImplementedError

    def run(self, row, dataset_root=None):
        assert dataset_root is not None
        model_input = self._load_model_input(row, dataset_root)
        model_output = self._run_model_inference(model_input)[0]
        output_path = self._save_model_output(model_output, row, dataset_root)
        row[self.output_col] = output_path
        return row

    def __repr__(self):
        return '{}Estimator(dir={}, input_col={}, output_col={}, checkpoint={})'.format(
            self.name, self.dir, self.input_col, self.output_col, self.checkpoint)
