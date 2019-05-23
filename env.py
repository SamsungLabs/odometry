import os
import sys

DATASET_PATH = "/dbstore/datasets/"
print(os.path.join(os.path.abspath(os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.pardir), 'submodules/tf_models/research/struct2depth'))
sys.path.append(os.path.join(os.path.abspath(os.pardir), 'submodules', 'tfoptflow/tfoptflow'))