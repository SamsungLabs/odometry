import numpy as np
import os

from slam.preprocessing.estimators import PWCNetEstimator
from slam.utils import resize_image, resize_image_arr


class BinocularDepthEstimator(PWCNetEstimator):

    def __init__(self, intrinsics, baseline_distance, batch_size=1, *args, **kwargs):
        self.batch_size = batch_size
        self.intrinsics = intrinsics
        self.baseline_distance = baseline_distance
        super().__init__(*args, **kwargs)
        self.name = 'BinocularDepth'

    def _convert_model_output_to_prediction(self, optical_flow):
        final_optical_flow = super()._convert_model_output_to_prediction(optical_flow)
        disparity = -final_optical_flow[..., 0] * self.intrinsics.width
        max_depth = self.intrinsics.f_x_scaled * self.baseline_distance
        depth = np.full_like(final_optical_flow[..., 0], max_depth).astype(float)
        depth[disparity > 0] = (self.intrinsics.f_x_scaled * self.baseline_distance) / disparity[disparity > 0]
        return depth.clip(max=max_depth)

    def _create_output_filename(self, row):
        filepath = row[self.input_col[0]]
        return '.'.join((os.path.splitext(os.path.basename(filepath))[0], self.ext))
