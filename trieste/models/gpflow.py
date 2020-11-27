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
from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar, Union, cast

import gpflow
import tensorflow as tf
from gpflow.models import GPModel, GPR, SGPR, SVGP, VGP
from trieste.type import ObserverEvaluations, QueryPoints, TensorType

from .. import utils
from .models import ProbabilisticModel, TrainableProbabilisticModel
from ..data import Dataset

Optimizer = Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer]


class CustomTrainable(tf.Module, TrainableProbabilisticModel, ABC):
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


class GPflowPredictor(ProbabilisticModel, ABC):
    """ A wrapper for a GPflow Gaussian process model. """

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """ The underlying GPflow model. """

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_f(query_points)

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        return self.model.predict_f_samples(query_points, num_samples)


_M = TypeVar("_M", bound="GaussianProcessRegression")


class GaussianProcessRegression(GPflowPredictor, CustomTrainable):
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

    def __deepcopy__(self: _M, memo: Dict[int, object]) -> _M:
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        cp = gpflow.utilities.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp


Batcher = Callable[[Dataset], Iterable[Tuple[tf.Tensor, tf.Tensor]]]
"""
Type alias for a function that creates minibatches from a :class:`~trieste.data.Dataset`.
"""


class SparseVariational(GPflowPredictor, TrainableProbabilisticModel):
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
        self._model = model
        self._data = data
        self._iterations = iterations
        self._optimizer = optimizer
        self._batcher = batcher
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
                self._model.training_loss_closure(batch), self._model.trainable_variables
            )

        batch_iterator = self._batcher(self._data)
        for i, batch in enumerate(batch_iterator):
            if i < self._iterations:
                _step(batch)

    def __deepcopy__(self, memo: Dict[int, object]) -> SparseVariational:
        deepcopied = SparseVariational(
            gpflow.utilities.deepcopy(self.model, memo),
            self._data,
            copy.deepcopy(self._optimizer, memo),
            self._iterations,
            cast(Batcher, copy.deepcopy(self._batcher, memo)),
            self._apply_jit
        )
        memo[id(self)] = deepcopied
        return deepcopied


class VariationalGaussianProcess(GaussianProcessRegression):
    def update(self, dataset: Dataset):
        model = self.model
        x, y = model.data
        assert dataset.query_points.shape[-1] == x.shape[-1]
        assert dataset.observations.shape[-1] == y.shape[-1]
        data = (dataset.query_points, dataset.observations)
        num_data = data[0].shape[0]

        f_mu, f_cov = self.model.predict_f(dataset.query_points, full_cov=True)  # [N, L], [L, N, N]
        assert self.model.q_sqrt.shape.ndims == 3

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrise q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to backtransform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(dataset.query_points, full_cov=True)  # [N, N]
        jitter_mat = gpflow.config.default_jitter() * tf.eye(num_data, dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data = data
        model.num_data = num_data
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_y(query_points)


supported_models: Dict[Any, Callable[[Any], CustomTrainable]] = {
    GPR: GaussianProcessRegression,
    SGPR: GaussianProcessRegression,
    VGP: VariationalGaussianProcess,
}
"""
:var supported_models: A mapping of third-party model types to :class:`CustomTrainable` classes
that wrap models of those types.
"""


@dataclass(frozen=True)
class ModelConfig:
    """ Specification for building a :class:`~trieste.models.TrainableProbabilisticModel`. """

    model: Union[tf.Module, TrainableProbabilisticModel]
    """ The :class:`~trieste.models.TrainableProbabilisticModel`, or the model to wrap in one. """

    optimizer: Optimizer = field(default_factory=lambda: gpflow.optimizers.Scipy())
    """ The optimizer with which to train the model (by minimizing its loss function). """

    optimizer_args: Dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the optimizer when training the model. """

    def __post_init__(self) -> None:
        self._check_model_type()

    @staticmethod
    def create_from_dict(d: Dict[str, Any]) -> ModelConfig:
        """
        :param d: A dictionary from which to construct this :class:`ModelConfig`.
        :return: A :class:`ModelConfig` built from ``d``.
        :raise TypeError: If the keys in ``d`` do not correspond to the parameters of
            :class:`ModelConfig`.
        """
        return ModelConfig(**d)

    def _check_model_type(self) -> None:
        if isinstance(self.model, TrainableProbabilisticModel):
            return

        for model_type in supported_models:
            if isinstance(self.model, model_type):
                return

        raise NotImplementedError(f"Not supported type {type(self.model)}")

    def create_model_interface(self) -> TrainableProbabilisticModel:
        """
        :return: A model built from this model configuration.
        """
        if isinstance(self.model, TrainableProbabilisticModel):
            return self.model

        for model_type, model_interface in supported_models.items():
            if isinstance(self.model, model_type):
                mi = model_interface(self.model)
                mi.set_optimizer(self.optimizer)
                mi.set_optimizer_args(self.optimizer_args)
                return mi

        raise NotImplementedError(f"Not supported type {type(self.model)}")


ModelSpec = Union[Dict[str, Any], ModelConfig, TrainableProbabilisticModel]
""" Type alias for any type that can be used to fully specify a model. """


def create_model(config: ModelSpec) -> TrainableProbabilisticModel:
    """
    :param config: A :class:`TrainableProbabilisticModel` or configuration of a model.
    :return: A :class:`~trieste.models.TrainableProbabilisticModel` build according to ``config``.
    """
    if isinstance(config, ModelConfig):
        return config.create_model_interface()
    elif isinstance(config, dict):
        return ModelConfig(**config).create_model_interface()
    elif isinstance(config, TrainableProbabilisticModel):
        return config
    raise NotImplementedError("Unknown format passed to create a TrainableProbabilisticModel.")
