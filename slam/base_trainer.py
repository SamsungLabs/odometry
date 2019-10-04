import os
import shutil
import mlflow
import datetime
import argparse
from keras.callbacks import ReduceLROnPlateau, TerminateOnNaN

import env

from slam.data_manager import GeneratorFactory
from slam.models import ModelFactory
from slam.evaluation import MlflowLogger, Predict, TerminateOnLR, ModelCheckpoint, CyclicLR
from slam.preprocessing import get_dataset_root, get_config, DATASET_TYPES
from slam.utils import set_computation, chmod


class BaseTrainer:
    def __init__(self,
                 leader_board,
                 run_name,
                 bundle_name,
                 cache=False,
                 batch_size=128,
                 epochs=100,
                 period=10,
                 save_best_only=False,
                 min_lr=1e-5,
                 reduce_factor=0.5,
                 no_cycle=False,
                 backend='numpy',
                 cuda=False,
                 per_process_gpu_memory_fraction=0.33,
                 use_mlflow=True,
                 seed=42,
                 stride=None,
                 min_frame_ind_diff=0,
                 max_frame_ind_diff=float('inf'),
                 **kwargs):

        self.tracking_uri = env.TRACKING_URI
        self.artifact_path = env.ARTIFACT_PATH
        self.project_path = env.PROJECT_PATH

        dataset_root = get_dataset_root(leader_board)
        self.config = get_config(dataset_root, leader_board, stride)

        self.dataset_root = dataset_root
        self.leader_board = leader_board
        self.run_name = run_name
        self.bundle_name = bundle_name
        self.cache = cache
        self.batch_size = batch_size
        self.epochs = epochs
        self.period = period
        self.save_best_only = save_best_only
        self.min_lr = min_lr
        self.reduce_factor = reduce_factor
        self.cyclic_lr = not no_cycle
        self.backend = backend
        self.cuda = cuda
        self.use_mlflow = use_mlflow
        self.seed = seed
        self.max_to_visualize = None

        self.construct_model_fn = None
        self.lr = None
        self.loss = None
        self.scale_rotation = None

        self.x_col = None
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = None
        self.load_mode = None
        self.preprocess_mode = None
        self.target_size = self.config['target_size']
        self.placeholder = None

        self.set_model_args()
        self.set_dataset_args()

        self.experiment_dir = None
        self.run_dir = None

        set_computation(self.seed, per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        self.client = None
        self.experiment = None

        self.min_frame_ind_diff = min_frame_ind_diff
        self.max_frame_ind_diff = max_frame_ind_diff

    def set_model_args(self):
        pass

    def set_dataset_args(self):
        pass

    def set_run_dir(self):
        self.run_dir = os.path.join(self.project_path, 'experiments', self.experiment_dir, self.run_name)

        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.makedirs(self.run_dir)

    def create_experiment(self):
        experiment_path = os.path.join(self.artifact_path, self.experiment_dir)
        os.makedirs(experiment_path)
        chmod(experiment_path)

        mlflow.create_experiment(self.leader_board, experiment_path)
        experiment = self.client.get_experiment_by_name(self.leader_board)
        return experiment

    def set_experiment(self):
        self.experiment_dir = self.leader_board.replace('/', '_')
        self.experiment = self.client.get_experiment_by_name(self.leader_board) or self.create_experiment()
        mlflow.set_experiment(self.leader_board)

    def get_run_data(self, run_name, leader_board):
        if leader_board == self.leader_board:
            experiment = self.experiment
        else:
            experiment = self.client.get_experiment_by_name(leader_board)

        filter_string = f'params.run_name = "{run_name}"'
        df = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=filter_string)

        if len(df):
            return df.iloc[0].to_dict()
        else:
            return None

    def start_run(self):
        if self.get_run_data(self.run_name, self.leader_board) is not None:
            raise RuntimeError(f'Run {self.run_name} already exists')

        mlflow.start_run(run_name=self.run_name)
        self.log_params()

    def log_params(self):
        mlflow.log_param('run_name', self.run_name)
        mlflow.log_param('bundle_name', self.bundle_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())
        mlflow.log_param('epochs', self.epochs)
        mlflow.log_param('seed', self.seed)
        mlflow.log_param('avg', False)

    def end_run(self):
        mlflow.log_metric('successfully_finished', 1)
        mlflow.end_run()

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
                                target_size=self.target_size,
                                x_col=self.x_col,
                                y_col=self.y_col,
                                image_col=self.image_col,
                                load_mode=self.load_mode,
                                batch_size=self.batch_size,
                                preprocess_mode=self.preprocess_mode,
                                depth_multiplicator=self.config['depth_multiplicator'],
                                cached_images={} if self.cache else None,
                                train_strides=self.config['train_strides'],
                                val_strides=self.config['val_strides'],
                                test_strides=self.config['test_strides'],
                                placeholder=self.placeholder,
                                min_frame_ind_diff=self.min_frame_ind_diff,
                                max_frame_ind_diff=self.max_frame_ind_diff)

    def get_model_factory(self, input_shapes):
        return ModelFactory(self.construct_model_fn,
                            input_shapes=input_shapes,
                            lr=self.lr,
                            loss=self.loss,
                            scale_rotation=self.scale_rotation)

    def get_callbacks(self, model, dataset, evaluate=True, save_dir=None, prefix=None):
        callbacks = []

        terminate_on_nan_callback = TerminateOnNaN()
        callbacks.append(terminate_on_nan_callback)

        save_dir = os.path.join(self.run_dir, save_dir or '.')
        monitor = 'val_RPE_t' if evaluate else 'val_loss'

        predict_callback = Predict(model=model,
                                   dataset=dataset,
                                   save_dir=save_dir,
                                   monitor=monitor,
                                   period=self.period,
                                   save_best_only=self.save_best_only,
                                   evaluate=evaluate,
                                   rpe_indices=self.config['rpe_indices'],
                                   max_to_visualize=self.max_to_visualize,
                                   backend=self.backend,
                                   cuda=self.cuda,
                                   workers=8)
        callbacks.append(predict_callback)

        if self.period:
            weights_dir = os.path.join(save_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            weights_filename = predict_callback.template + '.hdf5'
            weights_path = os.path.join(weights_dir, weights_filename)
            checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                                  filepath=weights_path,
                                                  save_best_only=self.save_best_only,
                                                  mode='min',
                                                  period=self.period)
            callbacks.append(checkpoint_callback)

        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=self.reduce_factor)
        callbacks.append(reduce_lr_callback)

        if self.cyclic_lr:
            lr_scheduler = CyclicLR(base_lr=self.lr * 0.1, max_lr=self.lr, step_size=1000, mode='exp_range')
            callbacks.append(lr_scheduler)

        terminate_on_lr_callback = TerminateOnLR(min_lr=self.min_lr)
        callbacks.append(terminate_on_lr_callback)

        if self.use_mlflow:
            mlflow_callback = MlflowLogger(alias={'loss': 'train_loss'},
                                           prefix=prefix,
                                           run_dir=self.run_dir,
                                           artifact_dir=self.run_name)
            callbacks.append(mlflow_callback)

        print('Training with callbacks:')
        for callback in callbacks:
            print(callback)
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
        if self.use_mlflow:
            self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)

            self.set_experiment()
            self.start_run()
            self.set_run_dir()

        dataset = self.get_dataset()

        model_factory = self.get_model_factory(dataset.input_shapes)
        model = model_factory.construct()
        print(model.summary())

        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.epochs,
                           evaluate=True)

        if self.use_mlflow:
            self.end_run()

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--leader_board', '--dataset_type', type=str,
                            choices=DATASET_TYPES, required=True)
        parser.add_argument('--run_name', '-n', type=str, required=True,
                            help='Name of the run. Must be unique and specific')
        parser.add_argument('--bundle_name', '-bn', type=str, required=True,
                            help='Name of the bundle. Must be unique and specific')
        parser.add_argument('--cache', action='store_true',
                            help='Cache inputs in RAM')
        parser.add_argument('--epochs', '-ep', type=int, default=100,
                            help='Number of epochs')
        parser.add_argument('--period', type=int, default=10,
                            help='Evaluate / checkpoint period'
                                 '(set to -1 for not saving weights and intermediate results)')
        parser.add_argument('--save_best_only', action='store_true',
                            help='Evaluate / checkpoint only if validation loss improves')
        parser.add_argument('--min_lr', type=float, default=1e-5,
                            help='Threshold value for learning rate in stopping criterion')
        parser.add_argument('--reduce_factor', type=float, default=0.5,
                            help='Reduce factor for learning rate')
        parser.add_argument('--no_cycle', action='store_true',
                            help='Disable cyclic learning rate')
        parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'],
                            help='Backend used for evaluation')
        parser.add_argument('--cuda', action='store_true',
                            help='Use GPU for evaluation (only for backend=="torch")')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
        parser.add_argument('--stride', type=int, default=None)
        parser.add_argument('--min_frame_ind_diff', type=int, default=0,
                            help='Minimum allowed stride between frames.')
        parser.add_argument('--max_frame_ind_diff', type=float, default=float('inf'),
                            help='Maximum allowed stride between frames.')

        return parser
