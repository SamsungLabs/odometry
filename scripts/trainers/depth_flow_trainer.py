import os
import functools
import mlflow
import argparse

import env
from odometry.data_manager import GeneratorFactory
from odometry.evaluation import PredictCallback
from odometry.models import ModelFactory, construct_depth_flow_model
from odometry.preprocessing.dataset_configs import get_config, DATASET_TYPES
from odometry.base_trainer import BaseTrainer


class DepthFlowTrainer(BaseTrainer):

    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 prediction_dir='predictions',
                 visuals_dir='visuals',
                 period=1,
                 save_best_only=False
                 ):
        super(DepthFlowTrainer, self).__init__(dataset_root=dataset_root,
                                               dataset_type=dataset_type,
                                               run_name=run_name,
                                               prediction_dir=prediction_dir,
                                               visuals_dir=visuals_dir,
                                               period=period,
                                               save_best_only=save_best_only)

    def train(self):

        config = get_config(self.dataset_root, self.dataset_type)

        #  All parameters
        epochs = 3
        mlflow.log_param('epochs', epochs)

        dataset = GeneratorFactory(
            csv_name='df.csv',
            dataset_root=self.dataset_root,
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

        construct_graph_fn = functools.partial(construct_depth_flow_model)
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
                                   predictions_dir=self.prediction_dir,
                                   visuals_dir=self.visuals_dir,
                                   period=self.period,
                                   save_best_only=self.save_best_only)

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

    parser = DepthFlowTrainer.get_parser()
    args = parser.parse_args()

    trainer = DepthFlowTrainer(dataset_root=args.dataset_root,
                               dataset_type=args.dataset_type,
                               run_name=args.run_name,
                               prediction_dir=args.prediction_dir,
                               visuals_dir=args.visuals_dir,
                               period=args.period,
                               save_best_only=args.save_best_only
                               )
    trainer.train()

