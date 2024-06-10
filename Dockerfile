# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install ffmpeg
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application into the container
COPY . .
