# Use the official CUDA image from NVIDIA as a base image
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Define the working directory
WORKDIR /app

RUN apt-get update && apt-get install libgl1

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install jaxlib==0.4.6

COPY . .

# Set the default command to run when starting the container
CMD ["/bin/bash"]
