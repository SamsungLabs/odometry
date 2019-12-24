import __init_path__

import mlflow
from sklearn.model_selection import train_test_split

from scripts.base_trainer import BaseTrainer
from slam.models import ModelWithConfidenceFactory


class ConfidenceTrainer(BaseTrainer):

    def __init__(self,
                 leader_board,
                 run_name,
                 bundle_name,
                 holdout=0.1,
                 confidence_epochs=100,
                 confidence_lr=0.001,
                 confidence_mode=None,
                 confidence_early_stopping=False,
                 **kwargs):

        self.holdout = holdout
        self.confidence_epochs = confidence_epochs
        self.confidence_mode = confidence_mode
        self.confidence_lr = confidence_lr
        self.confidence_early_stopping = confidence_early_stopping

        super().__init__(leader_board=leader_board,
                         run_name=run_name,
                         bundle_name=bundle_name,
                         **kwargs)

        self.max_to_visualize = 0
        self.placeholder = ['confidence']

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        return ModelWithConfidenceFactory(self.construct_model_fn,
                                          input_shapes=input_shapes,
                                          lr=self.lr,
                                          loss=self.loss,
                                          scale_rotation=self.scale_rotation,
                                          confidence_lr=self.confidence_lr,
                                          confidence_mode=self.confidence_mode)

    def train(self):
        if self.use_mlflow:
            self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)
            self.set_experiment()
            self.start_run()

        self.set_run_dir()

        dataset = self.get_dataset()

        train_index, confidence_index = train_test_split(dataset.df_train.index,
                                                         test_size=self.holdout,
                                                         random_state=42)
        df_train = dataset.df_train.loc[train_index]
        df_confidence = dataset.df_train.loc[confidence_index]

        dataset.df_train = df_train

        model_factory = self.get_model_factory(dataset.input_shapes)
        self.model = model_factory.construct()
        print(self.model.summary())

        self.fit_generator(model=self.model,
                           dataset=dataset,
                           epochs=self.epochs,
                           evaluate=False,
                           save_dir='dof')

        dataset.df_train = df_confidence

        self.cyclic_lr = False
        if self.confidence_early_stopping:
            self.early_stopping = True

        model = model_factory.freeze()
        self.fit_generator(model=model,
                           dataset=dataset,
                           epochs=self.confidence_epochs,
                           evaluate=False,
                           save_dir='confidence',
                           prefix='confidence',
                           save_metric='val_loss')

        if self.use_mlflow:
            self.end_run()

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--holdout', type=float, default=0.1,
                            help='Ratio of dataset to train confidence')
        parser.add_argument('--confidence_epochs', type=int, default=100,
                            help='Number of epochs to train confidence')
        parser.add_argument('--confidence_lr', type=float, default=0.001)
        parser.add_argument('--confidence_mode', type=str, default='log_std',
                            choices=[None, 'mse', 'mae', 'log_std', 'std'])
        parser.add_argument('--confidence_early_stopping', action='store_true')

        return parser
