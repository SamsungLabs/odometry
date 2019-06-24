import os
from pathlib import Path
import __init_path__
import env
import argparse
import mlflow
from odometry.data_manager import GeneratorFactory
import numpy as np
import pandas as pd
from functools import partial
import datetime


from odometry.models import ModelFactory, construct_flexible_model
from odometry.linalg import RelativeTrajectory
from odometry.evaluation import calculate_metrics, average_metrics
from odometry.utils import visualize_trajectory, visualize_trajectory_with_gt
import odometry.preprocessing.splits as configs


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
        records = list()
        for trajectory_id, indices in dataset.df_val.groupby(by='trajectory_id').indices.items():
            trajectory_id = trajectory_id.replace('/', '_')

            gt_trajectory = RelativeTrajectory.from_dataframe(dataset.df_val.iloc[indices]).to_global()
            predicted_trajectory = RelativeTrajectory.from_dataframe(predictions.iloc[indices]).to_global()

            predicted_trajectory.plot('plot_{}.html'.format(trajectory_id))

            metrics = calculate_metrics(gt_trajectory, predicted_trajectory, prefix='val')
            records.append(metrics)

            title = '{}: {}'.format(trajectory_id.upper(), metrics)
            visualize_trajectory(predicted_trajectory, title=title, is_3d=True,
                                 file_name='visualize_3d_{}.html'.format(trajectory_id))
            visualize_trajectory(predicted_trajectory, title=title, is_3d=False,
                                 file_name='visualize_2d_{}.html'.format(trajectory_id))

            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=True,
                                         file_name='visualize_3d_with_gt_{}.html'.format(trajectory_id))
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=False,
                                         file_name='visualize_2d_with_gt_{}.html'.format(trajectory_id))

        averaged_metrics = average_metrics(records, prefix="val_")
        mlflow.log_metrics(averaged_metrics, step=epoch)
        print(averaged_metrics)


def test(dataset, model):
    print('TEST')
    val_generator = dataset.get_val_generator()
    model_output = model.predict_generator(val_generator, steps=len(val_generator))
    predictions = pd.DataFrame(data=np.concatenate(model_output, 1),
                               index=dataset.df_val.index,
                               columns=dataset.y_col)

    print('EVALUATE')
    records = list()
    for trajectory_id, indices in dataset.df_val.groupby(by='trajectory_id').indices.items():
        trajectory_id = trajectory_id.replace('/', '_')

        gt_trajectory = RelativeTrajectory.from_dataframe(dataset.df_val.iloc[indices]).to_global()
        predicted_trajectory = RelativeTrajectory.from_dataframe(predictions.iloc[indices]).to_global()

        predicted_trajectory.plot('plot_{}.html'.format(trajectory_id))

        metrics = calculate_metrics(gt_trajectory, predicted_trajectory, prefix='val')
        records.append(metrics)

        title = '{}: {}'.format(trajectory_id.upper(), metrics)
        visualize_trajectory(predicted_trajectory, title=title, is_3d=True,
                             file_name='visualize_3d_{}.html'.format(trajectory_id))
        visualize_trajectory(predicted_trajectory, title=title, is_3d=False,
                             file_name='visualize_2d_{}.html'.format(trajectory_id))

        visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=True,
                                     file_name='visualize_3d_with_gt_{}.html'.format(trajectory_id))
        visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, is_3d=False,
                                     file_name='visualize_2d_with_gt_{}.html'.format(trajectory_id))

    averaged_metrics = average_metrics(records, prefix="test_")
    mlflow.log_metrics(averaged_metrics)
    print(averaged_metrics)


def get_config(dataset_type):

    if dataset_type == "kitti_1":
        config = configs.get_kitti_config_1()
        mlflow.set_experiment('kitti_1')
    elif dataset_type == "kitti_2":
        config = configs.get_kitti_config_2()
        mlflow.set_experiment('kitti_2')
    elif dataset_type == "discoman_iros_1":
        config = configs.get_discoman_iros_1_config()
        mlflow.set_experiment("discoman_iros_1")
    elif dataset_type == "discoman_debug":
        config = configs.get_discoman_debug_config()
        mlflow.set_experiment("discoman_debug")
    else:
        raise RuntimeError("Unexpected dataset type")

    return config


def main(dataset_root, dataset_type):

    config = get_config(dataset_type)

    mlflow.set_experiment(dataset_type)

    with mlflow.start_run(run_name='default_setup'):
        # All parameters
        epochs = 3
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("date", datetime.datetime.today().isoformat())
        target_size = (120, 160)

        dataset = GeneratorFactory(
            csv_name='df.csv',
            dataset_root=dataset_root,
            train_sequences=config['train_sequences'],
            val_sequences=config['val_sequences'],
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
            dataset_root=dataset_root,
            train_sequences=[],
            val_sequences=config['test_sequences'],
            target_size=target_size,
            x_col=['path_to_optical_flow'],
            image_columns=['path_to_optical_flow'],
            load_modes=['flow_xy'],
            preprocess_modes=['flow_xy'],
            val_sampling_step=2,
            cached_imgs={}
        )

        test(dataset, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path to dataset",
                        default='/dbstore/datasets/Odometry_team/discoman_v10_full/')
    parser.add_argument("--dataset_type", type=str, help="possible variants: kitti_1, kitti_2, discoman_iros_1,"
                                                         " discoman_debug , tum",
                        default='discoman_iros_1')

    args = parser.parse_args()
    main(args.dataset_root, args.dataset_type)