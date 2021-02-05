from dataclasses import astuple
from typing import Any, Mapping, Tuple

import gpflow
import gpflux
import numpy as np
import tensorflow as tf
from gpflow.conditionals.util import sample_mvn
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.type import TensorType
from gpflux.models import BayesianModel
from trieste.models.model_interfaces import _assert_data_is_compatible


class GPFluxModel(TrainableProbabilisticModel):

    def __init__(self,
                 model: BayesianModel,
                 data: Dataset,
                 num_epochs: int = 5000,
                 batch_size: int = 128,
                 **kwargs: Mapping[str, Any]):
        super().__init__()
        self._model = model
        self._data = data
        self.num_epochs = num_epochs
        self._batch_size = batch_size

    @property
    def model(self) -> BayesianModel:
        return self._model

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        (y_mean, y_var), (fs_mean, fs_var) = self.predict_y_fs(query_points)
        return y_mean, y_var

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """ Latent function samples """
        fs_mean, fs_var = self.predict_f(query_points)
        samples = sample_mvn(fs_mean, fs_var, False, num_samples=num_samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def update(self, dataset: Dataset) -> None:
        _assert_data_is_compatible(dataset, self._data)

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data

    def optimize(self, dataset: Dataset) -> None:
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
            )
        ]

        self._model.fit(x=dataset.query_points, y=dataset.observations,
                        batch_size=self._batch_size, epochs=self.num_epochs,
                        callbacks=callbacks)
