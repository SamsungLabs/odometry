# odometry

PREREQUISITE: 

Dependences on Ubuntu:
    libqglviewer-dev-qt4
    libeigen3-dev
    libsuitesparse-dev

Installation:
     1. conda env update --file conda.yml
     2. bash build.sh
         Or on cluster, you must
        a) ssh to any gpu node
        b) scl enable devtoolset-7 'bash build.sh'

Add odometry directory to PYTHONPATH: export PYTHONPATH=PATH-TO-ODOMETRY:$PYTHONPATH

For depth estimation download pretrained struct2depth weights from official site https://sites.google.com/view/struct2depth into weights directory

For OF estimation download pretrained tfoptflow weights from repo https://github.com/philferriere/tfoptflow

To dowload use git clone --recursive https://github.sec.samsung.net/AIMC-TSU/odometry.git

To run unittests: python -m unittest discover -s tests
