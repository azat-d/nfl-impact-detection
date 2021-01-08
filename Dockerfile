FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN rm /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-dev \
        python3-pip && \
    pip3 --no-cache-dir install \
        numpy==1.19.4 \
        ninja==1.10.0 \
        PyYAML==5.3.1 \
        mkl==2021.1.1 \
        mkl-include==2021.1.1 \
        setuptools==51.0.0 \
        cmake==3.18.4 \
        cffi==1.14.4 \
        typing_extensions==3.7.4.3 \
        future==0.18.2 \
        six==1.15.0 \
        requests==2.25.0 \
        dataclasses==0.8 \
        Pillow==8.0.1 \
        scipy==1.5.4 && \
    cd /tmp && \
    git clone -b v1.7.1 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    python3 setup.py install && \
    cd /tmp && \
    git clone -b v0.8.2 https://github.com/pytorch/vision.git && \
    cd vision && \
    python3 setup.py install && \
    cd / && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*