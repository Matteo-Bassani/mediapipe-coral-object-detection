import argparse
from settings import *
from libs import object_detector_extended
from core.mediapipe_object_detection_learning import train
from core.export import export
from hyperparameters.hyp_optimizer import optimize_hyperparameters


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, help='A boolean flag for hyperparameters optimization')
    args = parser.parse_args()

    # Check if the flag is provided without a value
    if args.hyperparams:
        # If hyperparams flag is set, optimize hyperparameters
        optimize_hyperparameters(args.hyperparams)
        return
    else:
        # If hyperparams flag is not set, train model
        train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
        validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)
        # Train model
        model, loss, coco_metrics = train(train_data, validation_data, DEF_HYP, WANDB)
        # Export model
        export(model, validation_data)
        return


if __name__ == "__main__":
    main()
