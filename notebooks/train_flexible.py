import os
from pathlib import Path
import __init_path__
import env
import mlflow
from odometry.utils import make_memory_safe
from odometry.preprocessing import parsers, estimators, prepare_trajectory
from odometry.data_manager import GeneratorFactory
import itertools
import numpy as np
import pandas as pd
from functools import partial

from odometry.models import ModelFactory, construct_flexible_model
from odometry.linalg import RelativeTrajectory
from odometry.evaluation import calculate_metrics
from odometry.utils import visualize_trajectory, visualize_trajectory_with_gt


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
    input_col = cols + [col + '_next' for col in cols]
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


def prepare_dataset(trajectories, target_size):

    single_frame_estimators, pair_frames_estimators = initialize_estimators(target_size)
    for trajectory in trajectories:

        parser = parsers.DISCOMANCSVParser(trajectory['name'], trajectory['json'])
        df = prepare_trajectory(trajectory['name'],
                                parser=parser,
                                single_frame_estimators=single_frame_estimators,
                                pair_frames_estimators=pair_frames_estimators,
                                stride=1)
        df.to_csv(os.path.join(trajectory['name'], 'df.csv'), index=False)


def train(dataset, model, epochs):

    for epoch in range(epochs):
        print('TRAIN')
        train_generator = dataset.get_train_generator()
        model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=1)

        print('VAL')
        val_generator = dataset.get_val_generator()
        model_output = model.predict_generator(val_generator, steps=len(val_generator))
        predictions = pd.DataFrame(data=np.concatenate(model_output, 1),
                                   index=dataset.df_val.index,
                                   columns=dataset.y_col)

        print('EVALUATE')
        for trajectory_id, indices in dataset.df_val.groupby(by='trajectory_id').indices.items():
            trajectory_id = trajectory_id.replace('/', '_')

            gt_trajectory = RelativeTrajectory.from_dataframe(dataset.df_val.iloc[indices]).to_global()
            predicted_trajectory = RelativeTrajectory.from_dataframe(predictions.iloc[indices]).to_global()

            predicted_trajectory.plot('plot_{}.html'.format(trajectory_id))

            metrics = calculate_metrics(gt_trajectory, predicted_trajectory, prefix='val')
            print(metrics)

            title = '{}: {}'.format(trajectory_id.upper(), metrics)
            visualize_trajectory(predicted_trajectory, title=title, is_3d=True,
                                 file_name='visualize_3d_{}.html'.format(trajectory_id))
            visualize_trajectory(predicted_trajectory, title=title, is_3d=False,
                                 file_name='visualize_2d_{}.html'.format(trajectory_id))

            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=True,
                                         file_name='visualize_3d_with_gt_{}.html'.format(trajectory_id))
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=False,
                                         file_name='visualize_2d_with_gt_{}.html'.format(trajectory_id))


def test(dataset, model):
    print('TEST')
    val_generator = dataset.get_val_generator()
    model_output = model.predict_generator(val_generator, steps=len(val_generator))
    predictions = pd.DataFrame(data=np.concatenate(model_output, 1),
                               index=dataset.df_val.index,
                               columns=dataset.y_col)

    print('EVALUATE')
    for trajectory_id, indices in dataset.df_val.groupby(by='trajectory_id').indices.items():
        trajectory_id = trajectory_id.replace('/', '_')

        gt_trajectory = RelativeTrajectory.from_dataframe(dataset.df_val.iloc[indices]).to_global()
        predicted_trajectory = RelativeTrajectory.from_dataframe(predictions.iloc[indices]).to_global()

        predicted_trajectory.plot('plot_{}.html'.format(trajectory_id))

        metrics = calculate_metrics(gt_trajectory, predicted_trajectory, prefix='val')
        print(metrics)

        title = '{}: {}'.format(trajectory_id.upper(), metrics)
        visualize_trajectory(predicted_trajectory, title=title, is_3d=True,
                             file_name='visualize_3d_{}.html'.format(trajectory_id))
        visualize_trajectory(predicted_trajectory, title=title, is_3d=False,
                             file_name='visualize_2d_{}.html'.format(trajectory_id))

        visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=True,
                                     file_name='visualize_3d_with_gt_{}.html'.format(trajectory_id))
        visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=False,
                                     file_name='visualize_2d_with_gt_{}.html'.format(trajectory_id))

def get_folds(dataset_root: Path):
    train = list()
    for d in dataset_root.joinpath('train').iterdir():
        traj = dict()
        traj['name'] = f'discoman/v10.5/train/{d.name}'
        traj['json'] = d.joinpath('0_traj.json')

        if traj['json'].exists():
            train.append(traj)
            break

    val = list()
    for d in dataset_root.joinpath('val').iterdir():
        traj = dict()
        traj['name'] = f'discoman/v10.5/val/{d.name}'
        traj['json'] = d.joinpath('0_traj.json')

        if traj['json'].exists():
            val.append(traj)
            break
        
    test = list()
    for d in dataset_root.joinpath('test').iterdir():
        traj = dict()
        traj['name'] = f'discoman/v10.5/test/{d.name}'
        traj['json'] = d.joinpath('0_traj.json')

        if traj['json'].exists():
            test.append(traj)
            break

    folds = {'train_sequences': train, 'val_sequences': val, 'test_sequences': test}
    return folds


if __name__ == '__main__':
    mlflow.set_experiment('discoman_v10.5')
    with mlflow.start_run(run_name='default_setup'):

        #All parameters
        epochs = 30
        mlflow.log_param("epochs", epochs)
        target_size = (120, 160)

        folds = get_folds(Path(env.DATASET_PATH).joinpath("renderbox/v10.5/traj/output"))

        prepare_dataset(itertools.chain(folds['train_sequences'], folds['val_sequences'], folds['test_sequences']),
                        target_size=target_size)

        dataset = GeneratorFactory(
            csv_name='df.csv',
            dataset_root='discoman',
            train_sequences=[i['name'] for i in folds['train_sequences']],
            val_sequences=[i['name'] for i in folds['val_sequences']],
            target_size=target_size,
            x_col=['path_to_optical_flow'],
            image_columns=['path_to_optical_flow'],
            load_modes=['flow_xy'],
            preprocess_modes=['flow_xy'],
            val_sampling_step=2,
            cached_imgs={}
        )

        construct_graph_fn = partial(construct_flexible_model, use_gated_convolutions=False)
        model_factory = ModelFactory(
            construct_graph_fn,
            input_shapes=dataset.input_shapes,
            lr=0.001,
            loss='mae',
            scale_rotation=50
        )
        model = model_factory.construct()

        train(dataset=dataset, model=model, epochs=epochs)

        dataset = GeneratorFactory(
            csv_name='df.csv',
            dataset_root='discoman',
            train_sequences=[],
            val_sequences=[i['name'] for i in folds['test_sequences']],
            target_size=target_size,
            x_col=['path_to_optical_flow'],
            image_columns=['path_to_optical_flow'],
            load_modes=['flow_xy'],
            preprocess_modes=['flow_xy'],
            val_sampling_step=2,
            cached_imgs={}
        )

