import numpy as np
from tqdm import trange
import pandas as pd

from slam.models import PretrainedModelFactory
from slam.preprocessing import PWCNetEstimator
from slam.utils import mlflow_logging
from slam.linalg import RelativeTrajectory


class BaseSlam:

    @mlflow_logging(prefix='slam.')
    def __init__(self, reloc_weights_path,
                 optflow_weights_path,
                 odometry_model_path,
                 optical_flow_shape,
                 knn=20,
                 rpe_indices='full',
                 loop_threshold=100):

        self.reloc_weights_path = reloc_weights_path
        self.optflow_weights_path = optflow_weights_path
        self.odometry_model_path = odometry_model_path
        self.knn = knn
        self.rpe_indices = rpe_indices
        self.loop_threshold = loop_threshold

        self.optical_flow_shape = optical_flow_shape

        self.reloc_model = None
        self.odometry_model = None
        self.optflow_model = None
        self.aggregator = None
        self.keyframe_selector = None

        self.last_frame = None
        self.last_keyframe = None

        self.frame_index = 0
        self.batch_size = 1

        self.odometry_mean_columns = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.odometry_std_columns = [col + '_confidence' for col in self.odometry_mean_columns]
        self.relocalization_columns = ['to_index', 'from_index', 'to_db_index', 'from_db_index']
        self.columns = self.odometry_mean_columns + self.odometry_std_columns + self.relocalization_columns

    def init(self, image):

        self.reloc_model.clear()
        self.aggregator.clear()

        self.last_frame = image
        self.last_keyframe = image

        self.reloc_model.add(self.last_keyframe, 0)
        self.frame_index = 1

    def construct(self):
        self.reloc_model = self.get_relocalization_model()
        self.optflow_model = self.get_optflow_model()
        self.odometry_model = self.get_odometry_model()
        self.aggregator = self.get_aggregator()
        self.keyframe_selector = self.get_keyframe_selector()

    def get_relocalization_model(self):
        raise RuntimeError('Not implemented')

    def get_optflow_model(self):
        return PWCNetEstimator(batch_size=self.batch_size,
                               input_col=[],
                               output_col=[],
                               sub_dir='',
                               checkpoint=self.optflow_weights_path)

    def get_aggregator(self):
        raise RuntimeError('Not implemented')

    def get_odometry_model(self):
        model_factory = self.get_odometry_model_factory()
        model = model_factory.construct()
        return model

    def get_odometry_model_factory(self):
        return PretrainedModelFactory(self.odometry_model_path)

    def get_keyframe_selector(self):
        raise RuntimeError('Not implemented')

    def predict_generator(self, generator):

        self.init(generator[0][0][0][0])

        frame_history = pd.DataFrame()

        for frame_index in trange(1, len(generator)):
            x, y = generator[frame_index]
            image = x[0][0]
            prediction = self.predict(image)
            frame_history = frame_history.append(prediction, ignore_index=True)

        paths = generator.df.path_to_rgb.values
        frame_history['to_path'] = paths[frame_history.to_index.values.astype('int')]
        frame_history['from_path'] = paths[frame_history.from_index.values.astype('int')]

        slam_trajectory = self.aggregator.get_trajectory()

        index_difference = (frame_history.to_index - frame_history.from_index)
        adjustment_measurements = frame_history[index_difference == 1].reset_index(drop=True)
        odometry_trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()

        self.aggregator.clear()
        self.aggregator.append(adjustment_measurements)
        is_loop_closure = index_difference > self.loop_threshold
        loop_closure = frame_history[is_loop_closure].reset_index(drop=True)
        self.aggregator.append(loop_closure)
        odometry_trajectory_with_loop_closures = self.aggregator.get_trajectory()

        return {'id': generator.trajectory_id,
                'slam_trajectory': slam_trajectory,
                'odometry_trajectory_with_loop_closures': odometry_trajectory_with_loop_closures,
                'odometry_trajectory': odometry_trajectory,
                'frame_history': frame_history}

    def predict(self, frame):

        prediction = pd.DataFrame(columns=self.columns)
        prediction = prediction.append(self.get_matches(frame))

        for index, row in prediction.iterrows():
            image_pair = self.get_image_pair(row, frame)
            optical_flow = self.optflow_model.predict(image_pair, target_size=self.optical_flow_shape)
            pose_mean, pose_std = self.predict_pose(optical_flow)
            prediction.loc[index, self.odometry_mean_columns] = pose_mean
            prediction.loc[index, self.odometry_std_columns] = pose_std

        self.last_frame = frame
        self.aggregator.append(prediction)
        self.frame_index += 1

        return prediction

    def get_matches(self, frame):
        matches = pd.DataFrame(columns=['to_index', 'from_index'], data=[[self.frame_index, self.frame_index - 1]])

        new_key_frame = self.keyframe_selector.is_key_frame(self.last_keyframe, frame, self.frame_index)
        if new_key_frame:
            matches = matches.append(self.reloc_model.predict(frame, self.frame_index), ignore_index=True)
            self.last_keyframe = frame
        return matches

    def get_image_pair(self, row, current_frame):
        if np.isnan(row['to_db_index']):
            batch = np.stack([self.last_frame, current_frame])
        else:
            batch = np.stack([self.reloc_model.images[int(row['from_db_index'])],
                              self.reloc_model.images[int(row['to_db_index'])]])
        return np.expand_dims(batch, axis=0)

    def predict_pose(self, optical_flow):
        model_output = np.stack(self.odometry_model.predict(optical_flow))
        mean = model_output[:, 0, 0]
        if model_output.shape[-1] == 2:
            std = model_output[:, 0, 1]
        else:
            std = np.ones_like(mean)
        return mean, std
