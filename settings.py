import os

DATASET = "car"
DEF_HYP = {
    'lr': 0.3,
    'epochs': 20,
    'batch_size': 32,
}
MAX_TRIALS = 30
STARTUP_TRIALS = 10
EARLY_STOPPING_PATIENCE = 5

# Settings
DATASETS_PATH = "datasets/"
DATASET_PATH = os.path.join(DATASETS_PATH, "dataset-"+DATASET)
DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "train")
DATASET_VAL_PATH = os.path.join(DATASET_PATH, "validation")
EXPORT_PATH = "exported_models/"
TFLITE_FLOAT32_NAME = "model.tflite"
TFLITE_INT8_NAME = "model_int8.tflite"
METADATA_PATH = os.path.join(EXPORT_PATH, "metadata.json")
METADATA_H_NAME = "metadata.hpp"
METADATA_H_PATH = os.path.join(EXPORT_PATH, METADATA_H_NAME)
