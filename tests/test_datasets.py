import unittest
import tests.__init_path__
import sys
import env
import os
import prepare_dataset.dataset_builder as db


env.DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'minidataset')


def prepare_builder(build_from) -> db.BaseDatasetBuilder:
    image_directory = None
    depth_directory = None
    video_path = None
    json_path = None
    csv_path = None
    txt_path = None
    dataset_builder = None
    sequence_directory = None

    weights_dir_path = os.path.join(env.PROJECT_PATH, 'weights')
    optical_flow_estimator_name = 'pwc'
    optical_flow_checkpoint = os.path.join(weights_dir_path, 'pwcnet.ckpt-595000')

    depth_estimator_name = 'struct2depth'
    depth_checkpoint = os.path.join(weights_dir_path, 'model-199160')
    # depth_estimator_name = 'senet'

    computation_kwargs = dict(
        cuda_visible_devices=1,
    )

    if build_from == db.VIDEO:
        sequence_directory = 'test_build_from_video'
        video_path = os.path.join(env.DATASET_PATH, 'saic_odometry/07/t_video5316726383492203313.mp4')
        dataset_builder = db.VideoDatasetBuilder

    if build_from == db.DIRECTORY:
        sequence_directory = 'test_build_from_dir'
        image_directory = os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/00/image_2')
        dataset_builder = db.ImagesDatasetBuilder

    if build_from == db.CSV:
        sequence_directory = 'test_build_from_csv'
        csv_path = ''
        dataset_builder = db.CSVDatasetBuilder

    if build_from == db.KITTI:
        sequence_directory = 'test_build_kitti'
        image_directory = os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/00/image_2')
        depth_directory = os.path.join(env.DATASET_PATH, 'KITTI_odometry_2012/dataset/sequences/00/depth_2')
        depth_estimator_name = None
        dataset_builder = db.KITTIDatasetBuilder

    if build_from == db.DISCOMAN:
        sequence_directory = 'test_build_discoman'
        json_path = os.path.join(env.DATASET_PATH, 'renderbox/v10.5/traj/output/train/0000/0_traj.json')
        depth_estimator_name = None
        dataset_builder = db.DISCOMANDatasetBuilder

    if build_from == db.TUM:
        sequence_directory = 'test_build_tum'
        txt_path = os.path.join(env.DATASET_PATH, 'tum_rgbd_flow/data/rgbd_dataset_freiburg2_coke/groundtruth.txt')
        dataset_builder = db.TUMDatasetBuilder

    estimate_optical_flow = optical_flow_estimator_name is not None
    estimate_depth = depth_estimator_name is not None

    builder = dataset_builder(sequence_directory,
                              build_from=build_from,
                              image_directory=image_directory,
                              depth_directory=depth_directory,
                              video_path=video_path,
                              json_path=json_path,
                              csv_path=csv_path,
                              txt_path=txt_path,
                              mode=dataset_builder.TEST,
                              estimate_optical_flow=estimate_optical_flow,
                              optical_flow_estimator_name=optical_flow_estimator_name,
                              optical_flow_checkpoint=optical_flow_checkpoint,
                              estimate_depth=estimate_depth,
                              depth_estimator_name=depth_estimator_name,
                              depth_checkpoint=depth_checkpoint,
                              memory_safe=True,
                              **computation_kwargs)

    return builder


class TestDatasets(unittest.TestCase):

    def test_tum(self) -> None:
        print("Started TUM test")
        builder = prepare_builder(db.TUM)
        builder.build()
        df = builder.dataframe

        for path in df["path_to_rgb"]:
            os.path.isfile(path)
        os.path.isfile(df['path_to_next_rgb'].iloc[-1])

        for path in df["path_to_depth"]:
            os.path.isfile(path)
        os.path.isfile(df['path_to_next_depth'].iloc[-1])

    # def test_discoman(self) -> None:
    #     print(sys.path)
    #     print("Started DISCOMAN test")
    #     builder = prepare_builder(db.DISCOMAN)
    #     builder.build()
    #     df = builder.dataframe
    #
    #     for path in df["path_to_rgb"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_rgb'].iloc[-1])
    #
    #     for path in df["path_to_depth"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_depth'].iloc[-1])
    #
    # def test_kitti(self) -> None:
    #     print(sys.path)
    #     print("Started KITTI test")
    #     builder = prepare_builder(db.KITTI)
    #     builder.build()
    #     df = builder.dataframe
    #
    #     for path in df["path_to_rgb"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_rgb'].iloc[-1])
    #
    #     for path in df["path_to_depth"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_depth'].iloc[-1])

    # def test_csv(self) -> None:
    #     print(sys.path)
    #     print("Started CSV test")
    #     builder = prepare_builder(db.KITTI)
    #     builder.build()
    #     df = builder.dataframe
    #
    #     for path in df["path_to_rgb"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_rgb'].iloc[-1])
    #
    #     for path in df["path_to_depth"]:
    #         os.path.isfile(path)
    #     os.path.isfile(df['path_to_next_depth'].iloc[-1])

    # def test_video(self) -> None:
    #     print(sys.path)
    #     print("Started VIDEO test")
    #     builder = prepare_builder(db.VIDEO)
    #     builder.build()
    #     df = builder.dataframe
    #
    #     for path in df["path_to_rgb"]:
    #         os.path.isfile(path)
    #
    #     for path in df["path_to_depth"]:
    #         os.path.isfile(path)