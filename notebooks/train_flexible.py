import os
from pathlib import Path
import __init_path__
import env

import itertools
import functools
import mlflow
import numpy as np
import pandas as pd

from odometry.data_manager import GeneratorFactory
from odometry.evaluation import PredictCallback
from odometry.models import ModelFactory, construct_flexible_model

from odometry.preprocessing import parsers, estimators, prepare_trajectory


def initialize_estimators(target_size):

    quaternion2euler_estimator = estimators.Quaternion2EulerEstimator(input_col=['q_w', 'q_x', 'q_y', 'q_z'],
                                                                      output_col=['euler_x', 'euler_y', 'euler_z'])

    depth_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                    'weights/model-199160')
    struct2depth_estimator = estimators.Struct2DepthEstimator(input_col='path_to_rgb',
                                                              output_col='path_to_depth',
                                                              sub_dir='depth',
                                                              checkpoint=depth_checkpoint,
                                                              height=target_size[0],
                                                              width=target_size[1])

    cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
    input_col = cols + [f'{col}_next' for col in cols]
    output_col = cols
    global2relative_estimator = estimators.Global2RelativeEstimator(input_col=input_col,
                                                                    output_col=output_col)

    optical_flow_checkpoint = '/Vol0/user/f.konokhov/tfoptflow/tfoptflow/tmp/pwcnet.ckpt-84000'
    # optical_flow_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # '/weights/pwcnet.ckpt-595000') # official weights
    pwcnet_estimator = estimators.PWCNetEstimator(input_col=['path_to_rgb', 'path_to_rgb_next'],
                                                  output_col='path_to_optical_flow',
                                                  sub_dir='optical_flow',
                                                  checkpoint=optical_flow_checkpoint)

    single_frame_estimators = [quaternion2euler_estimator, struct2depth_estimator]
    pair_frames_estimators = [global2relative_estimator, pwcnet_estimator]
    return single_frame_estimators, pair_frames_estimators


def prepare_dataset(trajectories, dataset_root, target_size):

    single_frame_estimators, pair_frames_estimators = initialize_estimators(target_size)
    for index, trajectory in enumerate(trajectories):
        parser = parsers.DISCOMANCSVParser(trajectory['csv'])
        df = prepare_trajectory(os.path.join(dataset_root, trajectory['name']),
                                parser=parser,
                                single_frame_estimators=single_frame_estimators,
                                pair_frames_estimators=pair_frames_estimators,
                                stride=1)
        path_to_save = os.path.join(dataset_root, trajectory['name'], 'df.csv')
        print(f'{index + 1} / {len(trajectories)}: saved trajectory(name={trajectory["name"]}, csv={trajectory["csv"]}) to {path_to_save}')
        df.to_csv(os.path.join(dataset_root, trajectory['name'], 'df.csv'), index=False)


def train(dataset, model, epochs, predictions_dir=None, visuals_dir=None, period=1, save_best_only=True):
    train_generator = dataset.get_train_generator()
    val_generator = dataset.get_val_generator()
    callback = PredictCallback(model,
                               dataset,
                               predictions_dir=predictions_dir,
                               visuals_dir=visuals_dir,
                               period=period,
                               save_best_only=save_best_only)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        shuffle=True,
        callbacks=[callback]
    )

        
def get_configs(dataset_root: Path):

    trajectories = list()
    for d in dataset_root.iterdir():
        traj = dict()
        traj['name'] = d.name
        traj['csv'] = d.joinpath('camera_gt.csv')

        if traj['csv'].exists():
            trajectories.append(traj)

    #how do we split trajectories?
    #train = trajectories[:100]
    #val = trajectories[100:150]
    #test = trajectories[150:]

    #this is a toy example
    train = trajectories[:1]
    val = trajectories[1:2]
    test = trajectories[2:3]

    configs = {'train_trajectories': train, 'val_trajectories': val, 'test_trajectories': test}
    return configs


if __name__ == '__main__':
    mlflow.set_experiment('discoman_v10_unzip')
    with mlflow.start_run(run_name='default_setup'):

        #All parameters
        epochs = 1
        mlflow.log_param('epochs', epochs)
        target_size = (120, 160)
        dataset_root = 'discoman'

        configs = get_configs(Path(env.DATASET_PATH).joinpath('Odometry_team/discoman_v10_unzip'))
        prepare_dataset(list(itertools.chain(configs['train_trajectories'], configs['val_trajectories'], configs['test_trajectories'])),
                        dataset_root=dataset_root,
                        target_size=target_size)

        dataset = GeneratorFactory(
            csv_name='df.csv',
            dataset_root=dataset_root,
            train_trajectories=[c['name'] for c in configs['train_trajectories']],
            val_trajectories=[c['name'] for c in configs['val_trajectories']],
            test_trajectories=[c['name'] for c in configs['test_trajectories']],
            target_size=target_size,
            x_col=['path_to_optical_flow'],
            image_col=['path_to_optical_flow'],
            load_mode=['flow_xy'],
            preprocess_mode=['flow_xy'],
            val_sampling_step=2,
            test_sampling_step=2,
            cached_images={}
        )

        construct_graph_fn = functools.partial(construct_flexible_model, use_gated_convolutions=False)
        model_factory = ModelFactory(
            construct_graph_fn,
            input_shapes=dataset.input_shapes,
            lr=0.001,
            loss='mae',
            scale_rotation=50
        )
        model = model_factory.construct()

        train(dataset=dataset,
              model=model,
              epochs=epochs,
              predictions_dir='predictions',
              visuals_dir='visuals',
              period=1,
              save_best_only=True
        )
