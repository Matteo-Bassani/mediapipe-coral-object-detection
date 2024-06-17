import optuna
from mediapipe_model_maker import quantization
from optuna.samplers import BaseSampler, TPESampler, RandomSampler

from settings import *
from libs import object_detector_extended
from core.train import train

WANDB = None
STUDY_NAME = "Study1"


def objective(trial):
    train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
    validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)
    lr = trial.suggest_categorical('lr', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])  # def 0.3
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])  # def 8
    epochs = trial.suggest_categorical('epochs', [1])  # def 10
    hyperparameters = {
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
    }
    global WANDB
    global STUDY_NAME

    model, loss, coco_metrics = train(train_data, validation_data, hyperparameters, WANDB)

    # Export 32-bit float model
    model.export_model(model_name='hyperparameters/' + STUDY_NAME + '/model_{}.tflite'.format(trial.number))

    # Perform post-training quantization (8-bit integer) and save quantized model
    quantization_config = quantization.QuantizationConfig.for_int8(
        representative_data=validation_data,
    )
    model.restore_float_ckpt()
    model.export_model(
        model_name='hyperparameters/' + STUDY_NAME + '/model_int8_{}.tflite'.format(trial.number),
        quantization_config=quantization_config,
    )

    # Get average precision and recall
    average_precision = coco_metrics.get('AP')
    average_recall = coco_metrics.get('ARmax100')
    f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall)
    return f1_score


def create_sampler(sampler_method) -> BaseSampler:
    if sampler_method == "random":
        sampler = RandomSampler()
    elif sampler_method == "tpe":
        sampler = TPESampler(n_startup_trials=STARTUP_TRIALS, multivariate=True)
    elif sampler_method == "skopt":
        from optuna.integration.skopt import SkoptSampler
        sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"Unknown sampler: {sampler_method}")
    return sampler


def optimize_hyperparameters(name="Study1", wandb_name=None):
    # Study name is initialized
    global STUDY_NAME
    STUDY_NAME = name

    # Wandb is initialized
    global WANDB
    WANDB = wandb_name

    # Path to databases is created
    path_databases = "hyperparameters/databases/"
    if not os.path.exists(path_databases):
        os.makedirs(path_databases)
    storage_name = "sqlite:///{}.db".format(path_databases + STUDY_NAME)

    path_models = "exported_models/hyperparameters/" + STUDY_NAME
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # Study is created
    study = optuna.create_study(study_name=STUDY_NAME,
                                storage=storage_name,
                                sampler=create_sampler("tpe"),
                                direction="maximize",
                                load_if_exists=True)

    # Prints command to launch optuna-dashboard
    dashboard_command = "optuna-dashboard " + storage_name
    print(dashboard_command)

    # Info about study are printed, dataframe is empty if the study is new
    info_study = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(info_study)

    # Study is optimized
    study.optimize(objective, n_trials=MAX_TRIALS)
    # Info about study are printed, dataframe is empty if the study is new
    best_metric_trial = study.best_trial

    # Best trial information
    print("Best trial according to metric:")
    print(f"   Number: {best_metric_trial.number}")
    print(f"  Value: {best_metric_trial.value}")
    print("  Params: ")
    for key, value in best_metric_trial.params.items():
        print(f"    {key}: {value}")
    print("  Metrics:")
    for key, value in best_metric_trial.user_attrs.items():
        print(f"    {key}: {value}")
