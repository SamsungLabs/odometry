import os
import shutil
import copy
import numpy as np

from utils.io_utils import load_image
from preprocessing.estimators.base_estimator import BaseEstimator


class NetworkEstimator(BaseEstimator):
    def __init__(self,
                 input_col,
                 output_col,
                 directory,
                 checkpoint):
        super(NetworkEstimator, self).__init__(input_col, output_col)
        
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        
        self.checkpoint = checkpoint
        self._load_model()

    def _load_model(self):
        pass

    def _convert_image_to_model_input(self, image):
        return np.array(image, dtype=np.float32)

    def _convert_model_output_to_prediction(self, output):
        return output

    def _load_model_input(self, row, dataset_root, input_col):
        return self._convert_image_to_model_input(
            load_image(os.path.join(dataset_root, row[input_col]))
        )
        return inputs
    
    def _save_model_output(self, model_output, row, dataset_root):
        output_path = self._create_output_path(row[self.input_col])
        np.save(os.path.join(dataset_root, output_path), 
                self._convert_model_output_to_prediction(model_output))

    def __repr__(self):
        return 'Estimator(dir={}, input_col={}, output_col={}, checkpoint={})'.format(
            self.directory, self.input_col, self.output_col, self.checkpoint)
