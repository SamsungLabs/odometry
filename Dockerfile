FROM nvidia/cuda:9.2-devel

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y wget bzip2 git
RUN apt-get install libqglviewer-dev

RUN mkdir /home/odometry
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/odometry/miniconda.sh
RUN bash /home/odometry/miniconda.sh -b -p /home/odometry/miniconda
RUN rm /home/odometry/miniconda.sh

ADD conda.yml conda.yml

RUN /home/odometry/miniconda/bin/conda env create -n odometry -f conda.yml

RUN apt-get install locales
RUN locale-gen en_US

RUN git clone https://github.com/uoip/g2opy.git g2opy
RUN cd g2opy
RUN mkdir build
RUN cd build
RUN cmake -DPYBIND11_PYTHON_VERSION=3.6 ..
RUN make -j8
RUN cd ..
RUN python setup.py install
