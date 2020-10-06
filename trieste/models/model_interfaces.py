# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod, ABC
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, Any

import gpflow
from gpflow.models import GPModel, GPR, SGPR, VGP, SVGP
import numpy as np
import tensorflow as tf

from .. import utils
from ..datasets import Dataset
from ..type import ObserverEvaluations, QueryPoints, TensorType


class ModelInterface(ABC):
    """ A trainable probabilistic model. """

    @abstractmethod
    def update(self, dataset: Dataset) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        """
        Return the predicted mean and variance of the latent function(s) at the specified
        ``query_points``, conditioned on the current data (see :meth:`update` to update the model
        given new data).

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        """
        Return ``num_samples`` samples from the predictive distribution at ``query_points``.

        :param query_points: The points at which to sample.
        :param num_samples: The number of samples at each point.
        :return: The samples. Has shape [S, Q, D], where S is the number of samples, Q is the number
            of query points, and D is the dimension of the predictive distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> None:
        """ Optimize the model parameters. """
        raise NotImplementedError


Optimizer = Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer]


class TrainableModelInterface(tf.Module, ModelInterface, ABC):
    """
    A utility class that provides a default optimization strategy, as well as the ability to modify
    various elements of this strategy.

    :cvar maxiter_default: The default maximum number of iterations to optimize the model for, when
        using a :class:`tf.optimizers.Optimizer`.
    :cvar create_optimizer_default: Builder for the default optimizer.
    :cvar apply_jit_function: If `True`, the default optimization procedure is compiled
        with :func:`tf.function`.
    """

    maxiter_default = 1000
    create_optimizer_default: Callable[[], Optimizer] = gpflow.optimizers.Scipy
    apply_jit_function = False

    @abstractmethod
    def loss(self) -> tf.Tensor:
        """ The training loss (to be minimized) on this model. """
        raise NotImplementedError

    @property
    def optimizer(self) -> Optimizer:
        """ The optimizer used to minimize the training loss. """
        if not hasattr(self, "_optimizer") or self._optimizer is None:
            self.set_optimizer(type(self).create_optimizer_default())
        return self._optimizer

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        :param optimizer: The optimizer to use.
        """
        self._optimizer = optimizer

    @property
    def optimizer_args(self) -> Dict[str, Any]:
        """ Keyword arguments passed to the optimizer during optimization. """
        if not hasattr(self, "_optimizer_args") or self._optimizer_args is None:
            self.set_optimizer_args(dict())
        return self._optimizer_args

    def set_optimizer_args(self, args: Dict[str, Any]) -> None:
        """
        :param args: The keyword arguments to use.
        """
        self._optimizer_args = args

    def optimize(self) -> None:
        if not hasattr(self, "_optimize_fn") or self._optimize_fn is None:
            self.set_optimize()
        return self._optimize_fn()

    def set_optimize(self, optimize_fn: Optional[Callable[[], None]] = None) -> None:
        """
        :param optimize_fn: The function to call on `optimize`. By default, constructs an
            optimization procedure from the current `loss`, `optimizer` and `optimizer_args`.
        """
        if optimize_fn is not None:
            self._optimize_fn = optimize_fn
            return

        if isinstance(self.optimizer, gpflow.optimizers.Scipy):

            def optimization_fn() -> None:
                @utils.jit(apply=self.apply_jit_function, autograph=True)
                def loss_fn() -> tf.Tensor:
                    return self.loss()

                trainables = self.model.trainable_variables
                return self.optimizer.minimize(loss_fn, variables=trainables, **self.optimizer_args)

        elif isinstance(self.optimizer, tf.optimizers.Optimizer):

            def optimization_fn() -> None:
                @utils.jit(apply=self.apply_jit_function, autograph=True)
                def loss_fn() -> tf.Tensor:
                    return self.loss()

                trainables = self.model.trainable_variables
                args = self.optimizer_args.copy()
                maxiter = args.pop("maxiter", self.maxiter_default)
                for _ in range(maxiter):
                    self.optimizer.minimize(loss_fn, var_list=trainables, **args)

        else:
            raise RuntimeError(
                f"Unknown type of optimizer ({type(self.optimizer)}) has been passed"
            )

        self._optimize_fn = optimization_fn


class GPflowPredictor(ModelInterface, ABC):
    """ A trainable wrapper for a GPflow Gaussian process model. """

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """ The underlying GPflow model. """

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_f(query_points)

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        return self.model.predict_f_samples(query_points, num_samples)


class GaussianProcessRegression(GPflowPredictor, TrainableModelInterface):
    def __init__(self, model: Union[GPR, SGPR]):
        """
        :param model: The GPflow model to wrap.
        """
        super().__init__()
        self._model = model

    @property
    def model(self) -> Union[GPR, SGPR]:
        return self._model

    def loss(self) -> tf.Tensor:
        return self._model.training_loss()

    def update(self, dataset: Dataset) -> None:
        x, y = self.model.data

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data = dataset.query_points, dataset.observations


Batcher = Callable[[Dataset], Iterable[Tuple[tf.Tensor, tf.Tensor]]]
"""
Type alias for a function that creates minibatches from a :class:`~trieste.datasets.Dataset`.
"""


class SparseVariational(GPflowPredictor, ABC):
    def __init__(
        self,
        model: SVGP,
        data: Dataset,
        optimizer: tf.optimizers.Optimizer,
        iterations: int,
        batcher: Batcher = lambda ds: [(ds.query_points, ds.observations)],
        apply_jit: bool = False,
    ):
        """
        :param model: The underlying GPflow sparse variational model.
        :param data: The initial training data.
        :param iterations: The number of iterations for which to optimize the model.
        :param optimizer: The optimizer to use for optimization.
        :param batcher: A function to convert training data into (mini)batches for optimization.
        """
        self._optimizer = optimizer
        self._iterations = iterations
        self._batcher = batcher
        self._model = model
        self._data = data
        self._apply_jit = apply_jit

    @property
    def model(self) -> SVGP:
        return self._model

    def update(self, dataset: Dataset) -> None:
        if dataset.query_points.shape[-1] != self._data.query_points.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != self._data.observations.shape[-1]:
            raise ValueError

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data

    def optimize(self) -> None:
        """
        Optimize the model in batches defined by the ``batcher`` argument to :meth:`__init__`.
        """

        @utils.jit(apply=self._apply_jit)
        def _step(batch: Tuple[tf.Tensor, tf.Tensor]) -> None:
            self._optimizer.minimize(
                lambda: self._model.training_loss(batch), self._model.trainable_variables
            )

        batch_iterator = self._batcher(self._data)
        for i, batch in enumerate(batch_iterator):
            if i < self._iterations:
                _step(batch)


class VariationalGaussianProcess(GaussianProcessRegression):
    def update(self, dataset: Dataset):
        model = self.model
        x, y = model.data
        assert dataset.query_points.shape[-1] == x.shape[-1]
        assert dataset.observations.shape[-1] == y.shape[-1]
        data = (dataset.query_points, dataset.observations)
        num_data = data[0].shape[0]
        num_latent_gps = model.num_latent_gps
        model.data = data
        model.num_data = num_data
        model.q_mu = gpflow.Parameter(np.zeros((num_data, num_latent_gps)))
        q_sqrt = np.eye(num_data)
        q_sqrt = np.repeat(q_sqrt[None], num_latent_gps, axis=0)
        model.q_sqrt = gpflow.Parameter(q_sqrt, transform=gpflow.utilities.triangular())

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_y(query_points)


supported_models: Dict[Any, Callable[[Any], TrainableModelInterface]] = {
    GPR: GaussianProcessRegression,
    SGPR: GaussianProcessRegression,
    VGP: VariationalGaussianProcess,
}
"""
:var supported_models: A mapping of third-party model types to :class:`ModelInterface` classes
that wrap models of those types.
"""
