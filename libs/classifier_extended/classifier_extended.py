from typing import Any, Callable, Optional, Sequence

import tensorflow as tf
from mediapipe_model_maker.python.core.tasks import classifier
from mediapipe_model_maker.python.core.data import classification_dataset as classification_ds
from mediapipe_model_maker.python.core.utils import model_util


class ClassifierExtended(classifier.Classifier):
    """An abstract base class that represents a TensorFlow classifier."""

    def _train_model(
            self,
            train_data: classification_ds.ClassificationDataset,
            validation_data: classification_ds.ClassificationDataset,
            callbacks: Sequence[tf.keras.callbacks.Callback],
            preprocessor: Optional[Callable[..., Any]] = None,
            checkpoint_path: Optional[str] = None,
    ):
        """Trains the classifier model.

    Compiles and fits the tf.keras `_model` and records the `_history`.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      preprocessor: An optional data preprocessor that can be used when
        generating a tf.data.Dataset.
      checkpoint_path: An optional directory for the checkpoint file to support
        continual training. If provided, loads model weights from the latest
        checkpoint in the directory.
    """
        tf.compat.v1.logging.info('Training the models...')
        if not self._hparams.repeat and len(train_data) < self._hparams.batch_size:
            raise ValueError(
                f"The size of the train_data {len(train_data)} can't be smaller than"
                f' batch_size {self._hparams.batch_size}. To solve this problem, set'
                ' the batch_size smaller or increase the size of the train_data.'
            )

        train_dataset = train_data.gen_tf_dataset(
            batch_size=self._hparams.batch_size,
            is_training=True,
            shuffle=self._shuffle,
            preprocess=preprocessor,
        )
        if self._hparams.repeat and self._hparams.steps_per_epoch is None:
            raise ValueError(
                '`steps_per_epoch` must be set if training `repeat` is True.'
            )
        self._hparams.steps_per_epoch = model_util.get_steps_per_epoch(
            steps_per_epoch=self._hparams.steps_per_epoch,
            batch_size=self._hparams.batch_size,
            train_data=train_data)
        if self._hparams.repeat:
            train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.take(count=self._hparams.steps_per_epoch)
        validation_dataset = validation_data.gen_tf_dataset(
            batch_size=self._hparams.batch_size,
            is_training=False,
            preprocess=preprocessor)
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss_function,
            metrics=self._metric_functions,
        )

        latest_checkpoint = (
            tf.train.latest_checkpoint(checkpoint_path)
            if checkpoint_path else None)
        if latest_checkpoint:
            print(f'Resuming from {latest_checkpoint}')
            self._model.load_weights(latest_checkpoint)

        # `steps_per_epoch` is intentionally set to None in case the dataset is not
        # repeated. Otherwise, the training process will stop when the dataset is
        # exhausted even if there are epochs remaining.
        if not self._hparams.repeat:
            steps_per_epoch = None
        else:
            steps_per_epoch = self._hparams.steps_per_epoch
        self._history = self._model.fit(
            x=train_dataset,
            epochs=self._hparams.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            callbacks=[callbacks],
            class_weight=self._hparams.class_weights,
        )
