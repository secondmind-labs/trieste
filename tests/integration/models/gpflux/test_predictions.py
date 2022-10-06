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

import numpy as np
import pytest

from tests.util.misc import hartmann_6_dataset, random_seed
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.objectives import Hartmann6


@pytest.mark.slow
@random_seed
@pytest.mark.parametrize("depth", [2, 3])
def test_dgp_model_close_to_actuals(depth: int) -> None:
    dataset_size = 50
    num_inducing = 50

    example_data = hartmann_6_dataset(dataset_size)

    dgp = build_vanilla_deep_gp(
        example_data,
        Hartmann6.search_space,
        depth,
        num_inducing,
        likelihood_variance=1e-5,
        trainable_likelihood=False,
    )
    model = DeepGaussianProcess(dgp)
    model.optimize(example_data)
    predicted_means, _ = model.predict(example_data.query_points)

    np.testing.assert_allclose(predicted_means, example_data.observations, atol=0.2, rtol=0.2)
