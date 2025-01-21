# Copyright 2021 The Trieste Contributors
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

"""
This file contains implementations of neural network architectures with Keras.
"""

from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import Any, Callable, Sequence

import dill
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

try:
    SafeModeScope = tf_keras.src.saving.serialization_lib.SafeModeScope
except AttributeError:  # pragma: no cover (tested but not by coverage)
    try:
        # in TF 2.14 it's in keras but not in tf_keras (but vice versa in TF 2.16)!
        from keras.src.saving.serialization_lib import SafeModeScope  # type:ignore[no-redef]
    except (ImportError, ModuleNotFoundError):
        SafeModeScope = contextlib.nullcontext
from tensorflow_probability.python.layers.distribution_layer import DistributionLambda, _serialize

from trieste.types import TensorType


class KerasEnsemble:
    """
    This class builds an ensemble of neural networks, using Keras. Individual networks must
    be instance of :class:`~trieste.models.keras_networks.KerasEnsembleNetwork`. This class
    is meant to be used with :class:`~trieste.models.keras_networks.DeepEnsemble` model wrapper,
    which compiles the model.
    """

    def __init__(
        self,
        networks: Sequence[KerasEnsembleNetwork],
    ) -> None:
        """
        :param networks: A list of neural network specifications, one for each member of the
            ensemble. The ensemble will be built using these specifications.
        :raise ValueError: If there are no objects in ``networks`` or we try to create
            a model with networks whose input or output shapes are not the same.
        """
        if not networks:
            raise ValueError(
                f"networks should consist of KerasEnsembleNetwork objects, however"
                f"received {networks} instead."
            )

        input_shapes, output_shapes, output_dtypes = [], [], []
        for index, network in enumerate(networks):
            network.network_name = f"model_{index}_"
            input_shapes.append(network.input_tensor_spec.shape)
            output_shapes.append(network.output_tensor_spec.shape)
            output_dtypes.append(network.output_tensor_spec.dtype)

        if not all(x == input_shapes[0] for x in input_shapes):
            raise ValueError(
                f"Input shapes for all networks must be the same, however"
                f"received {input_shapes} instead."
            )
        if not all(x == output_shapes[0] for x in output_shapes):
            raise ValueError(
                f"Output shapes for all networks must be the same, however"
                f"received {output_shapes} instead."
            )
        if not all(x == output_dtypes[0] for x in output_dtypes):
            raise ValueError(
                f"Output dtypes for all networks must be the same, however"
                f"received {output_dtypes} instead."
            )
        self.num_outputs = networks[0].flattened_output_shape
        self.output_dtype = networks[0].output_tensor_spec.dtype

        self._networks = networks

        self._model = self._build_ensemble()

    def __repr__(self) -> str:
        """"""
        return f"KerasEnsemble({self._networks!r})"

    @property
    def model(self) -> tf_keras.Model:
        """Returns built but uncompiled Keras ensemble model."""
        return self._model

    @property
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual neural
        network models in the ensemble.
        """
        return len(self._networks)

    def _build_ensemble(self) -> tf_keras.Model:
        """
        Builds the ensemble model by combining all the individual networks in a single Keras model.
        This method relies on ``connect_layers`` method of :class:`KerasEnsembleNetwork` objects
        to construct individual networks.

        :return: The Keras model.
        """
        inputs, outputs = zip(*[network.connect_layers() for network in self._networks])

        return tf_keras.Model(inputs=inputs, outputs=outputs)

    def __getstate__(self) -> dict[str, Any]:
        # When pickling use to_json to save the model.
        state = self.__dict__.copy()
        state["_model"] = self._model.to_json()
        state["_weights"] = self._model.get_weights()

        # Save the history callback (serializing any model)
        if self._model.history:
            history_model = self._model.history.model
            try:
                if history_model is self._model:
                    # no need to serialize the main model, just use a special value instead
                    self._model.history.model = ...
                elif history_model:
                    self._model.history.model = (
                        history_model.to_json(),
                        history_model.get_weights(),
                    )
                state["_history"] = dill.dumps(self._model.history)
            finally:
                self._model.history.model = history_model

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # When unpickling restore the model using model_from_json.
        self.__dict__.update(state)
        # TF 2.15 disallows loading lambdas without "safe-mode" being disabled
        # unfortunately, tfp.layers.DistributionLambda uses lambdas
        with SafeModeScope(False):
            self._model = tf_keras.models.model_from_json(
                state["_model"], custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL}
            )
        self._model.set_weights(state["_weights"])

        # Restore the history (including any model it contains)
        if "_history" in state:
            self._model.history = dill.loads(state["_history"])
            if self._model.history.model is ...:
                self._model.history.set_model(self._model)
            elif self._model.history.model:
                model_json, weights = self._model.history.model
                model = tf_keras.models.model_from_json(
                    model_json,
                    custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL},
                )
                model.set_weights(weights)
                self._model.history.set_model(model)


class KerasEnsembleNetwork:
    """
    This class is an interface that defines necessary attributes and methods for neural networks
    that are meant to be used for building ensembles by
    :class:`~trieste.models.keras_networks.KerasEnsemble`. Subclasses are not meant to
    build and compile Keras models, instead they are providing specification that
    :class:`~trieste.models.keras_networks.KerasEnsemble` will use to build the Keras model.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        network_name: str = "",
    ):
        """
        :param input_tensor_spec: Tensor specification for the input to the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param network_name: The name to be used when building the network.
        """
        if not isinstance(input_tensor_spec, tf.TensorSpec):
            raise ValueError(
                f"input_tensor_spec must be an instance of tf.TensorSpec, "
                f"received {type(input_tensor_spec)} instead."
            )
        if not isinstance(output_tensor_spec, tf.TensorSpec):
            raise ValueError(
                f"output_tensor_spec must be an instance of tf.TensorSpec, "
                f"received {type(output_tensor_spec)} instead."
            )

        self.input_tensor_spec = input_tensor_spec
        self.output_tensor_spec = output_tensor_spec
        self.network_name = network_name

    @property
    def input_layer_name(self) -> str:
        return self.network_name + "input"

    @property
    def output_layer_name(self) -> str:
        return self.network_name + "output"

    @property
    def flattened_output_shape(self) -> int:
        return int(np.prod(self.output_tensor_spec.shape))

    @abstractmethod
    def connect_layers(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Connects the layers of the neural network. Architecture, layers and layer specifications
        need to be defined by the subclasses.

        :return: Input and output tensor of the network, required by :class:`tf.keras.Model` to
            build a model.
        """
        raise NotImplementedError


class MultivariateNormalTriL(tfp.layers.MultivariateNormalTriL):  # type: ignore[misc]
    """Fixed version of tfp.layers.MultivariateNormalTriL that handles saving."""

    def __init__(
        self,
        event_size: int,
        convert_to_tensor_fn: Callable[
            [tfp.python.distributions.Distribution], TensorType
        ] = tfp.python.distributions.Distribution.sample,
        validate_args: bool = False,
        **kwargs: Any,
    ) -> None:
        self._event_size = event_size
        self._validate_args = validate_args
        super().__init__(event_size, convert_to_tensor_fn, validate_args, **kwargs)

    def get_config(self) -> dict[str, Any]:
        config = {
            "event_size": self._event_size,
            "validate_args": self._validate_args,
            "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
        }
        # skip DistributionLambda's get_config because we don't want to serialize the
        # make_distribution_fn: both to avoid confusing the constructor, and because it doesn't
        # seem to work in TF2.4.
        base_config = super(DistributionLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianNetwork(KerasEnsembleNetwork):
    """
    This class defines layers of a probabilistic neural network using Keras. The network
    architecture is a multilayer fully-connected feed-forward network, with Gaussian
    distribution as an output. The layers are meant to be built as an ensemble model by
    :class:`KerasEnsemble`. Note that this is not a Bayesian neural network.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        hidden_layer_args: Sequence[dict[str, Any]] = (
            {"units": 50, "activation": "relu"},
            {"units": 50, "activation": "relu"},
        ),
        independent: bool = False,
    ):
        """
        :param input_tensor_spec: Tensor specification for the input to the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param hidden_layer_args: Specification for building dense hidden layers. Each element in
            the sequence should be a dictionary containing arguments (keys) and their values for a
            :class:`~tf.keras.layers.Dense` hidden layer. Please check Keras Dense layer API for
            available arguments. Objects in the sequence will sequentially be used to add
            :class:`~tf.keras.layers.Dense` layers. Length of this sequence determines the number of
            hidden layers in the network. Default value is two hidden layers, 50 nodes each, with
            ReLu activation functions. Empty sequence needs to be passed to have no hidden layers.
        :param independent: In case multiple outputs are modeled, if set to `True` then
            :class:`~tfp.layers.IndependentNormal` layer
            is used as the output layer. This models outputs as independent, only the diagonal
            elements of the covariance matrix are parametrized. If left as the default `False`,
            then :class:`~tfp.layers.MultivariateNormalTriL` layer is used where correlations
            between outputs are learned as well.
        :raise ValueError: If objects in ``hidden_layer_args`` are not dictionaries.
        """
        super().__init__(input_tensor_spec, output_tensor_spec)

        self._hidden_layer_args = hidden_layer_args
        self._independent = independent

    def _gen_input_tensor(self) -> tf_keras.Input:
        input_tensor = tf_keras.Input(
            shape=self.input_tensor_spec.shape,
            dtype=self.input_tensor_spec.dtype,
            name=self.input_layer_name,
        )
        return input_tensor

    def _gen_hidden_layers(self, input_tensor: tf.Tensor) -> tf.Tensor:
        for index, hidden_layer_args in enumerate(self._hidden_layer_args):
            layer_name = f"{self.network_name}dense_{index}"
            layer = tf_keras.layers.Dense(
                **hidden_layer_args, name=layer_name, dtype=input_tensor.dtype.name
            )
            input_tensor = layer(input_tensor)

        return input_tensor

    def _gen_multi_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        dist_layer = tfp.layers.IndependentNormal if self._independent else MultivariateNormalTriL
        n_params = dist_layer.params_size(self.flattened_output_shape)

        parameter_layer = tf_keras.layers.Dense(
            n_params, name=self.network_name + "dense_parameters", dtype=input_tensor.dtype.name
        )(input_tensor)

        distribution = dist_layer(
            self.flattened_output_shape,
            tfp.python.distributions.Distribution.mean,
            name=self.output_layer_name,
            dtype=input_tensor.dtype.name,
        )(parameter_layer)

        return distribution

    def _gen_single_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        parameter_layer = tf_keras.layers.Dense(
            2, name=self.network_name + "dense_parameters", dtype=input_tensor.dtype.name
        )(input_tensor)

        def distribution_fn(inputs: TensorType) -> tfp.distributions.Distribution:
            return tfp.distributions.Normal(inputs[..., :1], tf.math.softplus(inputs[..., 1:]))

        distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=distribution_fn,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name=self.output_layer_name,
            dtype=input_tensor.dtype.name,
        )(parameter_layer)

        return distribution

    def connect_layers(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Connect all layers in the network. We start by generating an input tensor based on input
        tensor specification. Next we generate a sequence of hidden dense layers based on
        hidden layer arguments. Finally, we generate a dense layer whose nodes act as parameters of
        a Gaussian distribution in the final probabilistic layer.

        :return: Input and output tensor of the sequence of layers.
        """
        input_tensor = self._gen_input_tensor()
        hidden_tensor = self._gen_hidden_layers(input_tensor)

        if self.flattened_output_shape == 1:
            output_tensor = self._gen_single_output_layer(hidden_tensor)
        else:
            output_tensor = self._gen_multi_output_layer(hidden_tensor)

        return input_tensor, output_tensor
