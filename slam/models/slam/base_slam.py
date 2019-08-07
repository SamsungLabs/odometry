import numpy as np
import copy
from tqdm import trange
import pandas as pd

from tensorflow.python.client import device_lib

from slam.models import PretrainedModelFactory

from submodules.tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from submodules.tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

from slam.utils import mlflow_logging


class BaseSlam:

    @mlflow_logging(prefix='slam.')
    def __init__(self, reloc_weights_path,
                 optflow_weights_path,
                 odometry_model_path,
                 input_shapes,
                 knn=20,
                 rpe_indices='full'):

        self.reloc_weights_path = reloc_weights_path
        self.optflow_weights_path = optflow_weights_path
        self.odometry_model_path = odometry_model_path
        self.knn = knn
        self.rpe_indices = rpe_indices

        self.input_shapes = input_shapes

        self.reloc_model = None
        self.odometry_model = None
        self.optflow_model = None
        self.aggregator = None
        self.keyframe_selector = None

        self.frame_history = None

        self.last_frame = None
        self.last_keyframe = None

        self.frame_index = 0

    def get_relocalization_model(self):
        raise RuntimeError('Not implemented')

    def get_aggregator(self):
        raise RuntimeError('Not implemented')

    def get_keyframe_selector(self):
        raise RuntimeError('Not implemented')

    def get_optflow_model(self):
        nn_opts = copy.deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = True
        nn_opts['ckpt_path'] = self.optflow_weights_path
        nn_opts['batch_size'] = self.knn + 1

        devices = device_lib.list_local_devices()
        gpus = [device for device in devices if device.device_type == 'GPU']
        device = (gpus if len(gpus) else devices)[0].name

        nn_opts['gpu_devices'] = [device]
        nn_opts['controller'] = device
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
        return pwc_net(mode='test', options=nn_opts)

    def get_odometry_model_factory(self):
        return PretrainedModelFactory(self.odometry_model_path)

    def get_odometry_model(self):
        model_factory = self.get_odometry_model_factory()
        model = model_factory.construct()
        return model

    def batchify(self, df, current_frame):
        batch = np.zeros((self.knn + 1, 2, *self.input_shapes, 3))

        for index, row in df.iterrows():
            if np.isnan(row['to_db_index']):
                batch[index, 0] = current_frame
                batch[index, 1] = self.last_frame
            else:
                batch[index, 0] = self.reloc_model.images[int(row['to_db_index'])]
                batch[index, 1] = self.reloc_model.images[int(row['from_db_index'])]

        return batch

    @staticmethod
    def append_predict(matches, predicts):
        columns = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        for i, col in enumerate(columns):
            matches[col] = predicts[i][:len(matches), 0]
        return matches

    def construct(self):
        self.reloc_model = self.get_relocalization_model()

        self.optflow_model = self.get_optflow_model()

        self.odometry_model = self.get_odometry_model()

        self.aggregator = self.get_aggregator()

        self.keyframe_selector = self.get_keyframe_selector()

    def append_frame_path(self, matches, df):
        df_path = pd.DataFrame({'from_path': [df.rgb_path[i] for i in matches.from_index],
                                'to_path': [df.rgb_path[i] for i in matches.to_index]})
        return pd.concat([matches, df_path], axis=1)

    def predict_generator(self, generator):

        self.init(generator[0][0][0][0])

        for frame_index in trange(1, len(generator)):
            x, y = generator[frame_index]
            image = x[0][0]

            self.predict(image)

        frame_history = self.append_frame_path(self. frame_history, generator.df)

        return {'id': generator.trajectory_id,
                'trajectory': self.aggregator.get_trajectory(),
                'keyframe_history': self.keyframe_history,
                'frame_history': frame_history}

    def predict(self, frame):

        matches = pd.DataFrame({'to_db_index': [np.nan],
                                'from_db_index' : [np.nan],
                                'to_index': [self.frame_index],
                                'from_index': [self.frame_index - 1]})

        new_key_frame = self.keyframe_selector.is_key_frame(self.last_keyframe, frame, self.frame_index)

        if new_key_frame:
            matches = matches.append(self.reloc_model.predict(frame, self.frame_index))
            self.last_keyframe = frame

        batch = self.batchify(matches, frame)

        optflow = self.optflow_model.predict_from_img_pairs(batch, batch_size=self.knn + 1)

        predicts = self.odometry_model.predict(np.stack(optflow), batch_size=self.knn + 1)

        matches = self.append_predict(matches, predicts)

        self.frame_history = self.frame_history.append(matches)

        self.last_frame = frame

        self.aggregator.append(matches)

        self.frame_index += 1

    def init(self, image):
        self.frame_history = pd.DataFrame()

        self.reloc_model.clear()
        self.aggregator.clear()

        self.last_frame = image
        self.last_keyframe = image

        self.reloc_model.add(self.last_keyframe, 0)
        self.frame_index = 1
