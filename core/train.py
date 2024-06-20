import tensorflow as tf
from libs import object_detector_extended
from keras.callbacks import EarlyStopping
from settings import *
import wandb


def train(train_data, validation_data, hyperparameters, wandb_name, export_path):
    if wandb_name:
        # Wandb settings
        run = wandb.init(
            project=wandb_name,
            config={
                "learning_rate": hyperparameters['lr'],
                "batch_size": hyperparameters['batch_size'],
                "epochs": hyperparameters['epochs'],
            }
        )

    # Set Wandb Callback
    class WandbCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            wandb.log(logs)

    # Set early stopping callback
    early_stopping_callback = EarlyStopping(monitor='val_box_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    # Load pre-trained model and specify hyperparameters
    spec = object_detector_extended.SupportedModels.MOBILENET_V2_I320
    hparams = object_detector_extended.HParams(
        learning_rate=hyperparameters['lr'],
        batch_size=hyperparameters['batch_size'],
        epochs=hyperparameters['epochs'],
        export_dir=export_path,
    )
    options = object_detector_extended.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams,
    )

    if wandb_name:
        # Retrain model
        model = object_detector_extended.ObjectDetector.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options,
            callbacks=[WandbCallback(), early_stopping_callback]
        )
    else:
        # Retrain model
        model = object_detector_extended.ObjectDetector.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options,
            callbacks=[early_stopping_callback]
        )

    # Evaluate model performance
    loss, coco_metrics = model.evaluate(
        validation_data,
        batch_size=4,
    )

    if wandb_name:
        # Finish wandb run
        run.finish()

    # Print and return results
    print(f"Validation loss: {loss}")
    print(f"Validation metrics: {coco_metrics}")
    return model, loss, coco_metrics



