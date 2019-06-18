FROM nvidia/cuda:9.2-devel

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y wget bzip2

RUN mkdir /home/odometry
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/odometry/miniconda.sh
RUN bash /home/odometry/miniconda.sh -b -p /home/odometry/miniconda
RUN rm /home/odometry/miniconda.sh

ADD odometry.yml odometry.yml

RUN /home/odometry/miniconda/bin/conda env create -n odometry -f conda.yml

RUN apt-get install -y git