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

from typing import Dict, List, Union, Iterable

from ...data import Dataset
from ...type import TensorType
from ..optimizer import TrainingData
from .networks import KerasNetwork
from .utils import sample_with_replacement


class EnsembleDataTransformer:
    """
    A `DatasetTransformer` for Keras ensemble networks. Tensors in the data have to be
    carefully matched to names of input and output layers in the compiled model. Calling the class
    instance will also transform the data using the `transform_data` method implemented by each
    network in the ensemble. 

    Ensembles often use sampling with replacement of the inputs to reduce the generalization error.
    This can be done by initializing the `EnsembleDataTransformer` with the `bootstrap_data` set to `True`.
    """
    def __init__(self, networks: List[KerasNetwork], bootstrap_data: bool = False):
        """
        :param networks: A list of `KerasNetwork` objects that the compiled ensemble model uses.
        :param bootstrap_data: Resample data with replacement for training.
        """
        self._networks = networks
        self._bootstrap_data = bootstrap_data
        self.input_output_names: Dict[str, List[str]] = None

    def __call__(self, data: Dataset) -> TrainingData:
        """
        Transform data into inputs and outputs with correct names that can be used for training
        the Keras ensemble model. Note that it is crucial that the ensemble model populates `input_output_names` attribute with values after it has been compiled.    

        If the `EnsembleDataTransformer` is initialized with `bootstrap_data` set to `True`, data
        will be additionally resampled before transforming it using the `transform_data` method
        implemented by each network in the ensemble.

        :param dataset: A `Dataset` object consisting of `query_points` and `observations`. 
        :return: A `TrainingData` object.
        """
        inputs = {}
        outputs = {}

        assert len(self.input_output_names['inputs']) > 0 and len(self.input_output_names['outputs']) > 0 

        for index, network in enumerate(self._networks):

            input_name = self.input_output_names['inputs'][index]
            output_name = self.input_output_names['outputs'][index]

            if self._bootstrap_data:
                resampled_data = sample_with_replacement(data)
            else:
                resampled_data = data

            network_training_data = network.transform_data(resampled_data)
            network_query_points, network_observations = network_training_data.astuple()
            inputs[input_name] = network_query_points
            outputs[output_name] = network_observations

        return inputs, outputs

    def ensemblise_inputs(
        self,
        query_points: TensorType,
        transform_data: bool = False
    ) -> Union[TensorType, Iterable[TensorType]]:
        """
        Transform inputs to a correct input form for the ensemble, without necessarily transforming
        the input data. Useful for `predict` calls that operate only on inputs to the ensemble
        model. If `transform_data` is set to `True`, transformation will be executed using the 
        `transform_data` method implemented in each network.

        :param dataset: A `query_points` object, input part of `Dataset`. 
        :return: Return a new `query_points` object.
        """
        inputs = {}

        for index, network in enumerate(self._networks):

            input_name = self.input_output_names['inputs'][index]

            if transform_data:
                transformed_query_points = network.transform_data(query_points, True)
                inputs[input_name] = transformed_query_points
            else:
                inputs[input_name] = query_points

        return inputs
