from settings import *
from libs import object_detector_extended
from core.mediapipe_object_detection_learning import train
from core.export import export


def main():
    # Extract train and validation data
    train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
    validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)
    model, loss, coco_metrics = train(train_data, validation_data, DEF_HYP, WANDB)
    export(model, validation_data)


if __name__ == "__main__":
    main()
