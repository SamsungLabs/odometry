import os
import mlflow
import re

from scripts.base_trainer import BaseTrainer
from slam.models import PretrainedModelFactory
from slam.evaluation import Predict, MlflowLogger
from slam.preprocessing import DATASET_TYPES


class BaseTester(BaseTrainer):
    def __init__(self,
                 leader_board,
                 load_leader_board,
                 run_name,
                 load_name,
                 bundle_name,
                 cache=False,
                 epochs=0,
                 period=1,
                 checkpoint='final',
                 **kwargs):

        self.load_leader_board = load_leader_board
        self.load_name = load_name
        self.checkpoint = checkpoint
        self.pretrained_path = None
        self.epoch = 0

        assert self.checkpoint in ('best', 'final')

        super().__init__(leader_board=leader_board,
                         run_name=run_name,
                         bundle_name=bundle_name,
                         cache=False,
                         epochs=0,
                         period=1,
                         **kwargs)

    @staticmethod
    def get_value(data, key, default=None):
        value = data.get(key, default)
        try:
            value = eval(value)
        except:
            pass

        return value

    def get_pretrained_path(self, artifact_dir):
        weights_dir = os.path.join(artifact_dir, 'weights')
        print(f'Searching checkpoints in {weights_dir}')
        weights_filenames = sorted(os.listdir(weights_dir), key=self.sort_key)
        print('Sorting checkpoints...')
        pretrained_path = os.path.join(weights_dir, weights_filenames[0])
        print(f'Load checkpoint from {pretrained_path}')
        return pretrained_path

    def load_run(self):
        run_data = self.get_run_data(self.load_name, self.load_leader_board)
        if run_data is None:
            raise RuntimeError(f'Run {self.load_name} in {self.load_leader_board} not found')

        params = ['gen_factory.x_col', 'gen_factory.y_col', 'gen_factory.image_col',
                  'gen_factory.load_mode', 'gen_factory.preprocess_mode', 'gen_factory.placeholder',
                  'model.transform', 'model.channel_wise']

        for param_name in params:
            value = self.get_value(run_data, key=f'params.{param_name}')
            obj_name, attr_name = param_name.split('.')
            setattr(self, attr_name, value)
            print(f'Set {attr_name}:={value}')

        if self.predict_only:
            self.y_col = []
            self.evaluate = False

        self.agnostic = self.get_value(run_data, key='params.model.agnostic', default=True)

        self.model_name = self.get_value(run_data, key='params.model.name', default='Unknown')
        self.epoch = self.get_value(run_data, key='metrics.epoch', default=1)

        load_experiment_dir = self.load_leader_board.replace('/', '_')
        artifact_dir = os.path.join(self.artifact_path,
                                    load_experiment_dir,
                                    run_data['run_id'],
                                    'artifacts',
                                    self.load_name)
        self.pretrained_path = self.get_pretrained_path(artifact_dir)

    def start_run(self):
        self.load_run()
        super().start_run()

    def log_params(self):
        super().log_params()
        mlflow.log_param('model.name', self.model_name)
        mlflow.log_param('load_name', self.load_name)
        mlflow.log_param('load_leader_board', self.load_leader_board)
        mlflow.log_param('checkpoint', self.checkpoint)
        mlflow.log_param('pretrained_path', self.pretrained_path)
        mlflow.log_param('pretrained', 1)

    def sort_key(self, file_name):
        # Filename can be one of following:
        # {epoch, integer/"last"/"final"/"best"}_{metric name}_{metric value}
        # or
        # {epoch, integer/"last"/"final"/"best"}
        # or
        # {epoch, integer}-{metric value}

        base_name, ext = os.path.splitext(file_name)
        parts = re.split('_|-', base_name)

        epoch = parts[0]
        if self.checkpoint == 'best' and epoch == 'best':
            epoch = float('inf')
        elif self.checkpoint == 'final' and epoch in ('last', 'final'):
            epoch = float('inf') 
        elif epoch in ('best', 'final'):
            epoch = -1
        else:
            epoch = int(epoch)

        if len(parts) > 1:
            val_loss = float(parts[-1])
        else:
            val_loss = -float('inf')

        if self.checkpoint == 'best':
            key = (val_loss, -epoch)
        else:
            key = (-epoch, val_loss)

        print(f'{base_name} -> {parts} -> key {key}')
        return key

    def get_model_factory(self, input_shapes):
        return PretrainedModelFactory(pretrained_path=self.pretrained_path)

    def get_callbacks(self,
                      model,
                      dataset,
                      evaluate=True,
                      save_dir=None,
                      prefix=None,
                      save_metric='val_loss'):
        callbacks = []

        save_dir = os.path.join(self.run_dir, save_dir or '.')
        predict_callback = Predict(model=model,
                                   dataset=dataset,
                                   save_dir=save_dir,
                                   monitor=save_metric,
                                   period=1,
                                   evaluate=evaluate,
                                   save_best_only=False,
                                   rpe_indices=self.config['rpe_indices'],
                                   max_to_visualize=self.max_to_visualize,
                                   backend=self.backend,
                                   cuda=self.cuda,
                                   workers=8)
        predict_callback.epoch = self.epoch - 1
        predict_callback.template = self.checkpoint
        callbacks.append(predict_callback)

        if self.use_mlflow:
            mlflow_callback = MlflowLogger(alias={'loss': 'train_loss'},
                                           prefix=prefix,
                                           run_dir=self.run_dir,
                                           artifact_dir=self.run_name)
            mlflow_callback.epoch = self.epoch - 1
            callbacks.append(mlflow_callback)

        return callbacks

    def test(self):
        super().train()

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()

        parser.add_argument('--load_leader_board', '-load_dataset_type', type=str,
                            choices=DATASET_TYPES, required=True)
        parser.add_argument('--load_name', '-ln', type=str, required=True,
                            help='Name of the loaded run')
        parser.add_argument('--checkpoint', type=str, choices=['best', 'final'], default='final',
                            help='Which model weights to load')
        return parser
