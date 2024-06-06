import os

# Settings
DATASET_TRAIN_PATH = "datasettrain/"
DATASET_VAL_PATH = "datasetval/"
EXPORT_PATH = "exported_models/"
TFLITE_FLOAT32_NAME = "model.tflite"
TFLITE_INT8_NAME = "model_int8.tflite"
METADATA_PATH = "metadata.json"
METADATA_H_NAME = "metadata.hpp"
METADATA_H_PATH = os.path.join(EXPORT_PATH, METADATA_H_NAME)
WANDB = 1

DEF_HYP = {
    'lr': 0.3,
    'epochs': 10,
    'batch_size': 32,
}
MAX_TRIALS = 10
