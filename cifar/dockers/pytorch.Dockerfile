FROM nvcr.io/nvidia/pytorch:24.09-py3
LABEL maintainer="Konstantin Schürholt <konstantin.schuerholt@unisg.ch>"
# based on https://github.com/JulianAssmann/opencv-cuda-docker

RUN apt-get update 

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
    python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ca-certificates

# install othere ffcv dependencies
RUN pip3 install cupy-cuda12x numba

RUN apt-get update -y
RUN apt-get install -y libturbojpeg0-dev 
RUN apt-get install -y ffmpeg libsm6 libxext6 -y
RUN pip3 install opencv-python==4.8.0.74

# add further packages below.
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip3 install onnxruntime-training 
RUN pip3 install torch-ort
