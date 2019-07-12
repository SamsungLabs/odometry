import os
import mlflow
import datetime
import argparse
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

import env

from odometry.data_manager import GeneratorFactory
from odometry.models import ModelFactory
from odometry.evaluation import MLFlowLogger, Evaluate, TerminateOnLR
from odometry.preprocessing import get_config, DATASET_TYPES


class BaseTrainer:
    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 lsf=False,
                 batch=1,
                 epochs=100,
                 period=10,
                 save_best_only=False,
                 min_lr=1e-5,
                 reduce_factor=0.5):

        self.tracking_uri = env.TRACKING_URI
        self.artifact_path = env.ARTIFACT_PATH
        self.project_path = env.PROJECT_PATH

        self.config = get_config(dataset_root, dataset_type)

        self.start_run(self.config['exp_name'], run_name)

        mlflow.log_param('run_name', run_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())
        mlflow.log_param('epochs', epochs)

        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.run_name = run_name
        self.lsf = lsf
        self.batch = batch
        self.epochs = epochs
        self.period = period
        self.save_best_only = save_best_only
        self.min_lr = min_lr
        self.reduce_factor = reduce_factor

        self.set_model_args()

        self.set_dataset_args()

    def set_model_args(self):
        self.construct_model_fn = None
        self.lr = None
        self.loss = None
        self.scale_rotation = None

    def set_dataset_args(self):
        self.x_col = None
        self.image_col = None
        self.load_mode = None
        self.preprocess_mode = None

    def start_run(self, exp_name, run_name):
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        exp = client.get_experiment_by_name(exp_name)

        exp_dir = exp_name.replace('/', '_')
        if exp is None:
            exp_path = os.path.join(self.artifact_path, exp_dir)
            os.makedirs(exp_path, exist_ok=True)
            os.chmod(exp_path, 0o777)
            mlflow.create_experiment(exp_name, exp_path)

        run_names = list()
        for info in client.list_run_infos(exp.experiment_id):
            run_names.append(client.get_run(info.run_id).data.params.get('run_name', ''))

        if run_name in run_names:
            raise RuntimeError('run_name must be unique')

        self.run_dir = os.path.join(self.project_path, 'experiments', exp_dir, run_name)

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
                                image_col=self.image_col,
                                load_mode=self.load_mode,
                                preprocess_mode=self.preprocess_mode,
                                cached_images={})

    def get_model_factory(self, input_shapes):
        return ModelFactory(self.construct_model_fn,
                            input_shapes=input_shapes,
                            lr=self.lr,
                            loss=self.loss,
                            scale_rotation=self.scale_rotation)

    def get_callbacks(self, model, dataset):
        terminate_on_nan_callback = TerminateOnNaN()
        reduce_lr_callback = ReduceLROnPlateau(factor=self.reduce_factor)
        terminate_on_lr_callback = TerminateOnLR(min_lr=self.min_lr)
        mlflow_callback = MLFlowLogger()
        callbacks = [terminate_on_nan_callback,
                     reduce_lr_callback,
                     terminate_on_lr_callback,
                     mlflow_callback]

        if self.period:
            weights_dir = os.path.join(self.run_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, '{epoch:03d}-{val_loss:.6f}.hdf5')
            checkpoint_callback = ModelCheckpoint(filepath=weights_path,
                                                  save_best_only=self.save_best_only,
                                                  mode='min',
                                                  period=self.period)

            evaluate_callback = Evaluate(model=model,
                                         dataset=dataset,
                                         run_dir=self.run_dir,
                                         artifact_dir=self.run_name,
                                         period=self.period,
                                         save_best_only=self.save_best_only)
            callbacks.extend([checkpoint_callback,
                              evaluate_callback])

        return callbacks

    def fit_generator(self, model, dataset, epochs):
        train_generator = dataset.get_train_generator()
        val_generator = dataset.get_val_generator()
        callbacks = self.get_callbacks(model, dataset)

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
                           epochs=self.epochs)

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
