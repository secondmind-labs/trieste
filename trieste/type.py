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
"""This module contains type aliases."""
from typing import Union

import numpy as np
import tensorflow as tf

TensorType = Union[np.ndarray, tf.Tensor]
"""Type alias for tensor-like types."""

QueryPoints = TensorType
"""Type alias for points at which to query the observer."""

ObserverEvaluations = TensorType
"""Type alias for observed evaluations of the observer."""

AcquisitionEvaluations = TensorType
"""Type alias for evaluations of acquisition functions."""
