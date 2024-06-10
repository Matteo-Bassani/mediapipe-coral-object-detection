# MediaPipe Object Detection for Coral Dev Board Micro

- Run "python main.py" to train and export the model.
- Global settings can be changed in the "settings.py" file.

## Optional Arguments
- --hyperparams SampleStudy: start an hyperparameter study named "Study".
- --wandb SampleRun: log the training process to Weights & Biases under the name "SampleRun".

## Docker
- Build the Docker image with "docker build -t mediapipe-object-detection ."
- Run the Docker container with "docker run -it mediapipe-object-detection [CMD]"
 where [CMD] can be "python main.py" or any other command (/bin/bash to open a shell).

## Dataset
- The dataset should be placed in the "dataset" folder and divided into two subdirectories: "train" and "validation".
- The dataset directories should be in the Pascal VOC format, with images in "images" and annotations in "Annotations".
