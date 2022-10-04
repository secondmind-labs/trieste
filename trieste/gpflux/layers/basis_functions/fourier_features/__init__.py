#
# Copyright (c) 2021 The GPflux Contributors.
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
#
"""
A kernel's features for efficient sampling, used by
:class:`gpflux.sampling.KernelWithFeatureDecomposition`
"""

from trieste.gpflux.layers.basis_functions.fourier_features.multioutput.random import (
    MultiOutputRandomFourierFeatures,
    MultiOutputRandomFourierFeaturesCosine,
)
from trieste.gpflux.layers.basis_functions.fourier_features.random import (
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
    RandomFourierFeaturesCosine,
)

__all__ = [
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
    "RandomFourierFeaturesCosine",
    "MultiOutputRandomFourierFeatures",
    "MultiOutputRandomFourierFeaturesCosine",
]
