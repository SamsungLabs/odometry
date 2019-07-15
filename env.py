import os
import sys
import warnings
import mlflow

DATASET_PATH = '/dbstore/datasets/'
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PROJECT_PATH, 'submodules/tf_models/research/struct2depth'))
sys.path.append(os.path.join(PROJECT_PATH, 'submodules', 'tfoptflow/tfoptflow'))

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

TRACKING_URI = 'postgresql://odometry:YyXr9f8R@airulsf01/slamdb'
ARTIFACT_PATH = '/dbstore/datasets/robotics/mlflow/odometry/artifacts'
mlflow.set_tracking_uri(TRACKING_URI)

TUM_PATH = '/dbstore/datasets/Odometry_team/tum_rgbd/'
KITTI_PATH = '/dbstore/datasets/Odometry_team/KITTI_odometry_2012/'
DISCOMAN_V10_PATH = '/dbstore/datasets/Odometry_team/discoman_v10/'
SAIC_OFFICE_PATH = '/dbstore/datasets/Odometry_team/saic_office_prepare_v1_1_again/'
RETAIL_BOT_PATH = '/dbstore/datasets/Odometry_team/retail_bot/'
EUROC_PATH = '/dbstore/datasets/Odometry_team/EuRoC_prepare/'
ZJU_PATH = '/dbstore/datasets/Odometry_team/zju_prepare/'

