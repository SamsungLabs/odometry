from slam.base_trainer import BaseTrainer
from slam.data_manager.generator_factory import GeneratorFactory


class BaseSlamRunner(BaseTrainer):

    def __init__(self, knn=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reloc_weights = kwargs['reloc_weights']
        self.optflow_weights = kwargs['optflow_weights']
        self.odometry_model = kwargs['odometry_model']
        self.knn = knn

    def get_slam(self):
        raise RuntimeError('Not implemented')

    def get_dataset(self,
                    train_trajectories=None,
                    val_trajectories=None):

        train_trajectories = train_trajectories or self.config['train_trajectories']
        val_trajectories = val_trajectories or self.config['val_trajectories']
        test_trajectories = self.config['test_trajectories']
        self.x_col = ['path_to_rgb']
        self.image_col = ['path_to_rgb']
        self.load_mode = 'rgb'
        self.preprocess_mode = 'rgb'

        return GeneratorFactory(dataset_root=self.dataset_root,
                                train_trajectories=train_trajectories,
                                val_trajectories=val_trajectories,
                                test_trajectories=test_trajectories,
                                target_size=self.config['target_size'],
                                x_col=self.x_col,
                                image_col=self.image_col,
                                load_mode=self.load_mode,
                                preprocess_mode=self.preprocess_mode,
                                train_sampling_step=1,
                                batch_size=1,
                                cached_images={},
                                list_of_trajectory_generators=True)

    def run(self):

        dataset = self.get_dataset()

        slam = self.get_slam()
        slam.construct()

        slam.predict_generators(dataset.get_train_generator())
        slam.predict_generators(dataset.get_val_generator())
        slam.predict_generators(dataset.get_test_generator())

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--reloc_weights', type=str)
        parser.add_argument('--optflow_weights', type=str)
        parser.add_argument('--odometry_model', type=str)

        return parser
