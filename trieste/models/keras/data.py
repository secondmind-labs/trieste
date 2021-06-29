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
from typing import List

from trieste.data import Dataset
from trieste.models.keras import KerasNetwork
from trieste.models.optimizer import TrainingData


class EnsembleDataTransformer:
    """
    A `DatasetTransformer` for Keras ensemble networks.
    """
    def __init__(self, networks: List[KerasNetwork]):
        self._networks = networks

    def __call__(self, data: Dataset) -> TrainingData:
        inputs = {}
        outputs = {}

        for index, network in enumerate(self._networks):
            input_name = network.input_tensor_spec.name + "_" + str(index)
            if index == 0:
                output_name = 'dense'
            else:
                output_name = 'dense' + "_" + str(index)
            network_training_data = network.transform_training_data(data)
            network_query_points, network_observations = network_training_data.astuple()
            inputs[input_name] = network_query_points
            outputs[output_name] = network_observations

        return inputs, outputs
