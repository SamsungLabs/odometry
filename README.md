# odometry

PREREQUISITE: 

1. Install g2opy:
    - git clone https://github.com/uoip/g2opy.git submodules/g2opy
    - sudo apt install libqglviewer-dev
    - cd submodules/g2opy
    - mkdir build
    - cd build/
    - cmake -DPYBIND11_PYTHON_VERSION=3.6 ..
    - make -j8
    - cd ..
    - python setup.py install

Add odometry directory to PYTHONPATH: export PYTHONPATH=PATH-TO-ODOMETRY:$PYTHONPATH

For depth estimation download pretrained struct2depth weights from official site https://sites.google.com/view/struct2depth into weights directory

For OF estimation download pretrained tfoptflow weights from repo https://github.com/philferriere/tfoptflow

To dowload use git clone --recursive https://github.sec.samsung.net/AIMC-TSU/odometry.git

To run unittests: python -m unittest discover -s tests
