import os
import sys

DATASET_PATH = "/dbstore/datasets/"
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PROJECT_PATH, 'submodules/tf_models/research/struct2depth'))
sys.path.append(os.path.join(PROJECT_PATH, 'submodules', 'tfoptflow/tfoptflow'))