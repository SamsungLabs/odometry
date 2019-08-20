# odometry

PREREQUISITE: 

Dependences:
libqglviewer-dev-qt4
libeigen3-dev
libsuitesparse-dev

Installation:
1. build.sh

Add odometry directory to PYTHONPATH: export PYTHONPATH=PATH-TO-ODOMETRY:$PYTHONPATH

For depth estimation download pretrained struct2depth weights from official site https://sites.google.com/view/struct2depth into weights directory

For OF estimation download pretrained tfoptflow weights from repo https://github.com/philferriere/tfoptflow

To dowload use git clone --recursive https://github.sec.samsung.net/AIMC-TSU/odometry.git

To run unittests: python -m unittest discover -s tests
