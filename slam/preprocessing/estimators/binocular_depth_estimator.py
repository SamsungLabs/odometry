import numpy as np
import os

from slam.preprocessing.estimators import PWCNetEstimator
from slam.utils import resize_image, resize_image_arr
from slam.linalg import Intrinsics


class BinocularDepthEstimator(PWCNetEstimator):

    def __init__(self, batch_size=1, *args, **kwargs):
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        self.name = 'BinocularDepth'

    def _convert_model_output_to_prediction(self, optical_flow, row):
        final_optical_flow = super()._convert_model_output_to_prediction(optical_flow)

        intrinsics = Intrinsics(f_x=row[self.input_col[2]],
                                f_y=row[self.input_col[3]],
                                c_x=row[self.input_col[4]],
                                c_y=row[self.input_col[5]],
                                width=final_optical_flow.shape[1],
                                height=final_optical_flow.shape[0])

        baseline_distance = self.input_col[6]

        disparity = -final_optical_flow[..., 0] * intrinsics.width
        max_depth = intrinsics.f_x_scaled * baseline_distance
        depth = np.full_like(final_optical_flow[..., 0], max_depth).astype(float)
        depth[disparity > 0] = (intrinsics.f_x_scaled * baseline_distance) / disparity[disparity > 0]
        return depth.clip(max=max_depth)

    def _create_output_filename(self, row):
        filepath = row[self.input_col[0]]
        return '.'.join((os.path.splitext(os.path.basename(filepath))[0], self.ext))

    def run(self, row, dataset_root):
        model_input = self._load_model_input(row[self.input_col[0:2]], dataset_root)
        prediction = self.predict(model_input, row)
        output_path = self._save_model_prediction(prediction[0], row, dataset_root)
        row[self.output_col] = output_path
        return row

    def predict(self, batch, row):
        model_output = self._run_model_inference(batch)
        prediction = self._convert_model_output_to_prediction(model_output, row)
        return prediction
