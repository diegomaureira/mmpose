# Use NVIDIA CUDA 11.8 base image with Ubuntu 20.04 (Python 3.8 native)
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8, pip, git, build tools, and dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    pkg-config \
    ffmpeg \
    libavcodec-extra \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.2 with CUDA 11.8 support
RUN python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install OpenMMLab tools and dependencies
RUN python -m pip install -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.1.0" && \
    mim install mmdet

# Clone MMPose repository
# RUN git clone https://github.com/open-mmlab/mmpose.git
WORKDIR /mmpose

# Install Python dependencies for MMPose
#RUN python -m pip install -r requirements.txt

# Install MMPose in editable/development mode
#RUN python -m pip install -v -e .