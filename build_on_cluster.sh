#!/bin/bash
git clone https://github.com/uoip/g2opy.git submodules/g2opy
cd submodules/g2opy
mkdir build
cd build
scl enable devtoolset-7 'cmake -DPYBIND11_PYTHON_VERSION=3.6 ..'
scl enable devtoolset-7 'make -j8'
cd ..
python setup.py install