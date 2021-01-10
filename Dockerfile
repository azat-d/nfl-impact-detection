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
        setuptools==51.0.0 && \
    pip3 --no-cache-dir install \
        numpy==1.19.4 \
        ninja==1.10.0 \
        PyYAML==5.3.1 \
        mkl==2021.1.1 \
        mkl-include==2021.1.1 \
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
    export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" && \
    python3 setup.py install && \
    cd /tmp && \
    git clone -b v0.8.2 https://github.com/pytorch/vision.git && \
    cd vision && \
    FORCE_CUDA=1 python3 setup.py install && \
    cd / && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN cd /tmp && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout 3748b8b710a8603e966f0875dbb935f4b36ccaaa && \
    pip3 --no-cache-dir install \
        termcolor==1.1.0 \
        yacs==0.1.8 \
        tabulate==0.8.7 \
        cloudpickle==1.6.0 \
        matplotlib==3.3.3 \
        tqdm==4.54.1 \
        tensorboard==2.4.0 \
        fvcore==0.1.2.post20201213 \
        iopath==0.1.2 \
        pycocotools==2.0.2 \
        pydot==1.4.1 && \
    pip3 install . && \
    cd /tmp && \
    git clone https://github.com/facebookresearch/SlowFast.git && \
    cd SlowFast && \
    git checkout 3a8b0bd1f5de90bc843b694eaa0cc7d845640f42 && \
    pip3 --no-cache-dir install --upgrade pip && \
    pip3 --no-cache-dir install \
        av==8.0.2 \
        simplejson==3.17.2 \
        psutil==5.7.3 \
        pandas==1.1.5 \
        scikit-learn==0.24.0rc1 && \
    pip3 install . && \
    pip3 uninstall -y opencv-python && \
    cd / && \
    pip3 --no-cache-dir install \
        opencv-python-headless==4.4.0.46 \
        timm==0.3.2 && \
    rm -rf /tmp/*
