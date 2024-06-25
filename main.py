import warnings
warnings.filterwarnings("ignore")

import argparse
from settings import *
from libs import object_detector_extended
from core.train import train
from core.export import export
from core.hyp_optimizer import optimize_hyperparameters
from tools.extract_dataset import extract_dataset


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, help='Run hyperparameters optimization')
    parser.add_argument("--wandb", type=str, help='Log results to wandb')
    parser.add_argument("--dataset", type=str, help='Create a new dataset')
    parser.add_argument("--origin", type=str, help='Select the origin dataset')
    parser.add_argument("--object", type=str, help='Select the object class to extract')
    args = parser.parse_args()

    if args.dataset:
        if not args.origin or not args.object:
            print("Please provide the origin and object class arguments")
            return
        # Extract samples from fiftyone dataset
        extract_dataset(args.dataset, args.origin, args.object)
        return

    # Check if the flag is provided without a value
    if args.hyperparams:
        # If hyperparams flag is set, optimize hyperparameters
        optimize_hyperparameters(args.hyperparams, args.wandb)
        return
    else:
        # If dataset path doesn't exist, return error and exit
        if not os.path.exists(DATASET_PATH):
            print("Dataset path does not exist")
            return
        # If hyperparams flag is not set, train model
        train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
        validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)
        # Train model
        model, loss, coco_metrics = train(train_data, validation_data, DEF_HYP, args.wandb, TRAIN_EXPORT_PATH)
        # Export model
        export(model, validation_data, TRAIN_EXPORT_PATH)
        return


if __name__ == "__main__":
    main()
