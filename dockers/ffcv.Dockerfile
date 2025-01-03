FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Konstantin Schürholt <konstantin.schuerholt@unisg.ch>"
# based on https://github.com/JulianAssmann/opencv-cuda-docker

RUN apt-get update 

# FFCV requires opencv, which is not available in the pytorch docker image.
# Installing via apt messes up cuda/mpi, so we need to build opencv from source.
# -> build opencv from source, with cuda (DWITH_CUDA=ON)
# -> pkg_config needs to be generated for ffcv to find opencv (DOPENCV_GENERATE_PKGCONFIG=YES) #note, can be "ON" depending on opencv version

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.7.0

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
RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget --no-check-certificate https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget --no-check-certificate https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -D BUILD_opencv_python_bindings_generator=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
	    -DOPENCV_GENERATE_PKGCONFIG=YES \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        ..
# Make
WORKDIR /opt/opencv-${OPENCV_VERSION}/build
RUN make && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

RUN pip3 install torch torchvision torchaudio
# install othere ffcv dependencies
RUN pip3 install cupy-cuda12x numba

RUN apt-get update -y
RUN apt-get install -y libturbojpeg0-dev 
RUN apt-get install -y ffmpeg libsm6 libxext6 -y
RUN pip3 install opencv-python==4.8.0.74
# install ffcv
RUN pip3 install ffcv

# add further packages below.
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip3 install onnxruntime-training 
RUN pip3 install torch-ort
