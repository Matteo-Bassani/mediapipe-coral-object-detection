# Use the official CUDA image from NVIDIA as a base image
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Define the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    software-properties-common

# Aggiungi la chiave GPG per il repository Coral Edge TPU
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Aggiungi il repository Coral Edge TPU alla lista delle sorgenti
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN apt-get update && apt-get install -y  \
    libgl1 \
    edgetpu-compiler

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install jaxlib==0.4.6

COPY . .

# Set the default command to run when starting the container
CMD ["/bin/bash"]
