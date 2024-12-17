FROM nvcr.io/nvidia/pytorch:24.11-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libtbb2 \
        libpostproc-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install lightning

WORKDIR /workspace/