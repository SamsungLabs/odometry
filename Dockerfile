FROM nvidia/cuda:9.2-devel

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y wget bzip2

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p ~/miniconda
RUN rm ~/miniconda.sh

ADD odometry.yml odometry.yml

RUN ~/miniconda/bin/conda env create -n odometry -f odometry.yml