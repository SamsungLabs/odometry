import os
import mlflow
import datetime
import argparse
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

import env

from slam.data_manager import GeneratorFactory
from slam.models import ModelFactory
from slam.evaluation import MlflowLogger, Predict, TerminateOnLR
from slam.preprocessing import get_config, DATASET_TYPES
from slam.utils import set_computation


class BaseTrainer:
    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 seed=42,
                 lsf=False,
                 cache=False,
                 batch=1,
                 epochs=100,
                 period=10,
                 save_best_only=False,
                 min_lr=1e-5,
                 reduce_factor=0.5,
                 **kwargs):

        self.tracking_uri = env.TRACKING_URI
        self.artifact_path = env.ARTIFACT_PATH
        self.project_path = env.PROJECT_PATH

        self.config = get_config(dataset_root, dataset_type)

        self.start_run(self.config['exp_name'], run_name)

        mlflow.log_param('run_name', run_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('seed', seed)

        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.run_name = run_name
        self.seed = seed
        self.lsf = lsf
        self.cache = cache
        self.batch = batch
        self.epochs = epochs
        self.period = period
        self.save_best_only = save_best_only
        self.min_lr = min_lr
        self.reduce_factor = reduce_factor
        self.max_to_visualize = 5

        self.set_model_args()

        self.set_dataset_args()

        set_computation(self.seed)

    def set_model_args(self):
        self.construct_model_fn = None
        self.lr = None
        self.loss = None
        self.scale_rotation = None

    def set_dataset_args(self):
        self.x_col = None
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = None
        self.load_mode = None
        self.preprocess_mode = None
        self.batch_size = 128

    def start_run(self, exp_name, run_name):
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        exp = client.get_experiment_by_name(exp_name)

        exp_dir = exp_name.replace('/', '_')
        if exp is None:
            exp_path = os.path.join(self.artifact_path, exp_dir)
            os.makedirs(exp_path, exist_ok=True)
            os.chmod(exp_path, 0o777)
            mlflow.create_experiment(exp_name, exp_path)
            exp = client.get_experiment_by_name(exp_name)

        run_names = list()
        for info in client.list_run_infos(exp.experiment_id):
            run_names.append(client.get_run(info.run_id).data.params.get('run_name', ''))

        if run_name in run_names:
            raise RuntimeError('run_name must be unique')

        self.run_dir = os.path.join(self.project_path, 'experiments', exp_dir, run_name)
        self.save_dir = self.run_dir

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=run_name)

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        train_trajectories = train_trajectories or self.config['train_trajectories']
        val_trajectories = val_trajectories or self.config['val_trajectories']
        test_trajectories = self.config['test_trajectories']
        return GeneratorFactory(dataset_root=self.dataset_root,
                                train_trajectories=train_trajectories,
                                val_trajectories=val_trajectories,
                                test_trajectories=test_trajectories,
                                target_size=self.config['target_size'],
                                x_col=self.x_col,
                                y_col=self.y_col,
                                image_col=self.image_col,
                                load_mode=self.load_mode,
                                batch_size=self.batch_size,
                                preprocess_mode=self.preprocess_mode,
                                depth_multiplicator=self.config['depth_multiplicator'],
                                cached_images={} if self.cache else None)

    def get_model_factory(self, input_shapes):
        return ModelFactory(self.construct_model_fn,
                            input_shapes=input_shapes,
                            lr=self.lr,
                            loss=self.loss,
                            scale_rotation=self.scale_rotation)

    def get_callbacks(self, model, dataset, evaluate=True, save_dir=None, prefix=None):
        terminate_on_nan_callback = TerminateOnNaN()
        reduce_lr_callback = ReduceLROnPlateau(factor=self.reduce_factor)
        terminate_on_lr_callback = TerminateOnLR(min_lr=self.min_lr, prefix=prefix)
        mlflow_callback = MlflowLogger(prefix=prefix)
        callbacks = [terminate_on_nan_callback,
                     reduce_lr_callback,
                     terminate_on_lr_callback,
                     mlflow_callback]

        if self.period:
            save_dir = os.path.join(self.run_dir, save_dir) if save_dir else self.run_dir
            weights_dir = os.path.join(save_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            weights_filename = '{epoch:03d}_train:{loss:.6f}_val:{val_loss:.6f}.hdf5'
            weights_path = os.path.join(weights_dir, weights_filename)
            checkpoint_callback = ModelCheckpoint(filepath=weights_path,
                                                  save_best_only=self.save_best_only,
                                                  mode='min',
                                                  period=self.period)
            callbacks.append(checkpoint_callback)

        predict_callback = Predict(model=model,
                                   dataset=dataset,
                                   run_dir=self.run_dir,
                                   save_dir=save_dir,
                                   artifact_dir=self.run_name,
                                   prefix=prefix,
                                   period=self.period,
                                   save_best_only=self.save_best_only,
                                   evaluate=evaluate,
                                   rpe_indices=self.config['rpe_indices'],
                                   max_to_visualize=self.max_to_visualize)
        callbacks.append(predict_callback)

        return callbacks

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None, prefix=None):
        train_generator = dataset.get_train_generator()
        val_generator = dataset.get_val_generator()
        callbacks = self.get_callbacks(model,
                                       dataset,
                                       evaluate=evaluate,
                                       save_dir=save_dir,
                                       prefix=prefix)

        model.fit_generator(train_generator,
                            steps_per_epoch=len(train_generator),
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=len(val_generator),
                            shuffle=True,
                            callbacks=callbacks)

    def train(self):
        dataset = self.get_dataset()

        model_factory = self.get_model_factory(dataset.input_shapes)
        model = model_factory.construct()

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs,
                           evaluate=True,
                           save_dir='.')

        mlflow.log_metric('successfully_finished', 1)
        mlflow.end_run()

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset_root', '-r', type=str, required=True,
                            help='Directory with trajectories')
        parser.add_argument('--dataset_type', '-t', type=str,
                            choices=DATASET_TYPES, required=True)
        parser.add_argument('--run_name', '-n', type=str, required=True,
                            help='Name of the run. Must be unique and specific')
        parser.add_argument('--cache', action='store_true',
                            help='Cache inputs in RAM')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
        parser.add_argument('--epochs', '-ep', type=int, default=100,
                            help='Number of epochs')
        parser.add_argument('--period', type=int, default=10,
                            help='Evaluate / checkpoint period (set to -1 for not saving weights and intermediate results)')
        parser.add_argument('--save_best_only', action='store_true',
                            help='Evaluate / checkpoint only if validation loss improves')
        parser.add_argument('--min_lr', type=float, default=1e-5,
                            help='Threshold value for learning rate in stopping criterion')
        parser.add_argument('--reduce_factor', type=int, default=0.5,
                            help='Reduce factor for learning rate')

        return parser
