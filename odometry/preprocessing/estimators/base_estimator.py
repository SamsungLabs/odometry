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

    def _create_output_filename(self, row):
        input_col = self.input_col if isinstance(self.input_col, str) else self.input_col[-1]
        input_path = row[input_col]
        if input_path is None:
            return None

        output_filename = '.'.join((os.path.splitext(os.path.basename(input_path))[0], self.ext))
        return output_filename

    def _extend(self, row, values):
        for key, value in zip(self.output_col, values):
            row[key] = value
        return row

    def run(self, row, dataset_root=None):
        pass

    def __repr__(self):
        return 'Estimator(input_col={}, output_col={})'.format(self.input_col, self.output_col)
