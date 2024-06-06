import optuna
from settings import *
from libs import object_detector_extended
from core.mediapipe_object_detection_learning import train


def objective(trial):
    train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
    validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)
    lr = trial.suggest_float('lr', 0.001, 0.1)
    batch_size = trial.suggest_int('batch_size', 8, 32, log=True)
    epochs = trial.suggest_int('epochs', 10, 50)
    hyperparameters = {
        'lr': lr,
        'epochs': batch_size,
        'batch_size': epochs,
    }
    _, loss, coco_metrics = train(train_data, validation_data, hyperparameters, 0)
    return coco_metrics.get('AP')


study = optuna.create_study()
study.optimize(objective, n_trials=MAX_TRIALS)
