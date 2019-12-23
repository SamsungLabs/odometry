import os
import numpy as np
import pandas as pd
import cv2

from .network_estimator import NetworkEstimator
from slam.utils import load_image, undistort_image


class UndistortionEstimator(NetworkEstimator):

    def __init__(self, *args, **kwargs):
        kwargs = dict(checkpoint=None, **kwargs)
        super().__init__(name='Undistortion',
                         *args,
                         **kwargs)

    def _load_model(self):
        pass

    def run(self, row: pd.Series, dataset_root: str):
        image_filepath = row[self.input_col[0]]
        image = load_image(os.path.join(dataset_root, image_filepath))

        K = np.array(row[self.input_col[1]])
        D = np.array(row[self.input_col[2]])
        R = np.array(row[self.input_col[3]])
        P = np.array(row[self.input_col[4]])

        undistorted_image = undistort_image(image, K=K, D=D, R=R, P=P)

        os.makedirs(os.path.join(dataset_root, self.dir), exist_ok=True)
        output_path = os.path.join(self.dir, os.path.basename(image_filepath))
        cv2.imwrite(os.path.join(dataset_root, output_path), undistorted_image)

        row[self.output_col] = output_path
        return row
