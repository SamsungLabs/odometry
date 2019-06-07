import os
import shutil
import copy
import numpy as np


class BaseEstimator:
    def __init__(self,
                 input_col,
                 output_col):
        self.input_col = input_col
        self.output_col = output_col
        self.ext = 'npy'

    def _construct_filename(self, reference_path):
        if reference_path is None:
            return None
        if isinstance(reference_path, list):
            reference_path = reference_path[-1]
        filename = '.'.join((os.path.splitext(os.path.basename(reference_path))[0], self.ext))
        return filename
    
    def _create_output_path(self, input_path):
        if input_path is None:
            return None
        path_to_save = os.path.join(self.directory, self._construct_filename(input_path))
        return path_to_save

    def run(self, row, dataset_root=None):
        pass

    def __repr__(self):
        return 'Estimator(dir={}, input_col={}, output_col={})'.format(
            self.directory, self.input_col, self.output_col)
