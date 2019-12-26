# Deep_SLAM (STAR-MOS)

SLAM based on deep learning odometry for camera position estimation.

## Goals	
1) Share results from paper "Training Deep SLAM on Single Frames" [[arxiv](https://arxiv.org/abs/1912.05405)]. 
2) Share framework for training, evaluating and storing results of various odometry models.

## Maintainers (Max. 3)	
* Anna Vorontsova (a.vorontsova@samsung.com)
* Igor Slynko (i.slynko@samsung.com)

## Committers	
* Anna Vorontsova (a.vorontsova@samsung.com)
* Dmitry Zhukov (d.zhukov@samsung.com)
* Igor Slynko (i.slynko@samsung.com)

## Getting Started
### Prerequisites
- conda 
- rest of  requirements listed in `conda.yml`

### How to build
`conda install -f  conda.yml`

### How to test
python -m unittest discover -s tests
