# Odometry

SLAM based on deep learning odometry for camera position estimation.

## Goals	
1) Share results from paper "Training Deep SLAM on Single Frames" [[arxiv](https://arxiv.org/abs/1912.05405)]. 
2) Share framework for training, evaluating and storing results of various odometry models.

## Getting Started
### Prerequisites
- libqglviewer-dev-qt4
- libeigen3-dev
- libsuitesparse-dev
- conda 
- rest of  requirements listed in `conda.yml`

### How to build
git clone --recursive https://github.sec.samsung.net/AIMC-TSU/odometry.git
1. conda env update --file conda.yml<br>
2. conda activate odometry
3. bash build.sh<br>
  Or on cluster, you must<br>
  a) ssh to any gpu node<br>
  b) scl enable devtoolset-7 'bash build.sh'
4. Update PYTHONPATH: export PYTHONPATH=path_to_this_repo:$PYTHONPATH
5. (optional) For depth estimation download pretrained struct2depth weights from official site https://sites.google.com/view/struct2depth into weights directory
6. (optional) For OF estimation download pretrained tfoptflow weights from repo https://github.com/philferriere/tfoptflow


### How to test
python -m unittest discover -s tests
