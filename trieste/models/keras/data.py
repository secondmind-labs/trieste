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
from typing import Dict, List

from trieste.data import Dataset
from trieste.models.keras.networks import KerasNetwork
from trieste.models.optimizer import TrainingData


class EnsembleDataTransformer:
    """
    A `DatasetTransformer` for Keras ensemble networks.
    """
    def __init__(self, networks: List[KerasNetwork]):
        self._networks = networks
        self.input_output_names: Dict[str, List[str]] = None

    def __call__(self, data: Dataset) -> TrainingData:
        inputs = {}
        outputs = {}

        assert len(self.input_output_names['inputs']) > 0 and len(self.input_output_names['outputs']) > 0 

        for index, network in enumerate(self._networks):

            input_name = self.input_output_names['inputs'][index]
            output_name = self.input_output_names['outputs'][index]

            network_training_data = network.transform_training_data(data)
            network_query_points, network_observations = network_training_data.astuple()
            inputs[input_name] = network_query_points
            outputs[output_name] = network_observations

        return inputs, outputs

    # @property
    # def input_output_names(self):
    #     return self.__input_output_names

    # @input_output_names.setter
    # def set_input_output_names(self, names = None) -> None:
    #     self.__input_output_names = names
