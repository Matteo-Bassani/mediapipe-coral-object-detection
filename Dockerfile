# Use the official CUDA image from NVIDIA as a base image
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Define the working directory
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the default command to run when starting the container
CMD ["/bin/bash"]
