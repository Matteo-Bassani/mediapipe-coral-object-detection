# Use the official CUDA image from NVIDIA as a base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3.8 -m pip install --upgrade pip

# Verify the installation
RUN python3.8 --version && pip3 --version

# Define the working directory
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the default command to run when starting the container
CMD ["/bin/bash"]
