import functools
import mlflow
import argparse

import __init_path__
import env
from odometry.data_manager import GeneratorFactory
from odometry.evaluation import PredictCallback
from odometry.models import ModelFactory, construct_flexible_model
from odometry.preprocessing.dataset_configs import get_config, DATASET_TYPES
from odometry.utils import str2bool


def train(dataset_root: str,
          dataset_type: str,
          run_name: str,
          predictions_dir: str = None,
          visuals_dir: str = None,
          period: int = 1,
          save_best_only: bool = True
          ) -> None:
    """
    This is script that shows how to use our small framework. You're welcome to modify it and experiment with new models.

    :param dataset_root:
    :param dataset_type:
    :param run_name:
    :param predictions_dir:
    :param visuals_dir:
    :param period:
    :param save_best_only:
    :return:
    """

    config = get_config(dataset_root, dataset_type)

    # MLFlow initialization
    mlflow.set_experiment(config['exp_name'])
    mlflow.start_run(run_name=run_name)

    #  All parameters
    mlflow.log_param('run_name', run_name)
    epochs = 2
    mlflow.log_param('epochs', epochs)

    dataset = GeneratorFactory(
        csv_name='df.csv',
        dataset_root=dataset_root,
        train_trajectories=config['train_trajectories'],
        val_trajectories=config['val_trajectories'],
        test_trajectories=config['test_trajectories'],
        target_size=config['target_size'],
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

    train_generator = dataset.get_train_generator()
    val_generator = dataset.get_val_generator()
    val_steps = len(val_generator) if val_generator else None
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
        validation_steps=val_steps,
        shuffle=True,
        callbacks=[callback]
    )

    mlflow.end_run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', '-r', type=str, help='Directory with trajectories', required=True)
    parser.add_argument('--dataset_type', '-t', type=str, choices=DATASET_TYPES, required=True)
    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Must be unique and specific',
                        required=True)
    parser.add_argument('--prediction_dir', '-p', type=str, help='Name of subdir to store predictions',
                        default='predictions')
    parser.add_argument('--visuals_dir', '-v', type=str, help='Name of subdir to store visualizations',
                        default='visuals')
    parser.add_argument('--period', type=int, help='Period of evaluating train and val metrics',
                        default=1)
    parser.add_argument('--save_best_only', type=str2bool, help='Evaluate metrics only for best losses',
                        default=False)
    args = parser.parse_args()

    train(dataset_root=args.dataset_root,
          dataset_type=args.dataset_type,
          run_name=args.run_name,
          predictions_dir=args.prediction_dir,
          visuals_dir=args.visuals_dir,
          period=args.period,
          save_best_only=args.save_best_only
          )
