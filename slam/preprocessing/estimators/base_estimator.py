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
        self.name = 'Base'

    def _add_output(self, row, values):
        for key, value in zip(self.output_col, values):
            row[key] = value
        return row

    def _drop_input(self, row):
        for key in self.input_col:
            del row[key]
        return row

    def run(self, row, dataset_root=None):
        pass

    def __repr__(self):
        return '{}Estimator(input_col={}, output_col={})'.format(self.name, self.input_col, self.output_col)
