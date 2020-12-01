# %% [markdown]
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

# %% [markdown]
# # Inequality constraints: constrained optimization

# %%
import gpflow
from dataclasses import astuple
from gpflow import set_trainable, default_float
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import trieste

from util import inequality_constraints_utils as util

# %%
np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## The problem
#
# In this tutorial, we replicate one of the results of Gardner, 2014 [1], specifically their synthetic experiment "simulation 1", which consists of an objective function with a single constraint, defined over a two-dimensional input domain. We'll start by defining the problem parameters.

# %%
class Sim:
    threshold = 0.5

    @staticmethod
    def objective(input_data):
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)
        return z[:, None]

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None]

search_space = trieste.space.Box(
    tf.cast([0.0, 0.0], default_float()), tf.cast([6.0, 6.0], default_float())
)

# %% [markdown]
# The objective and constraint functions are accessible as methods on the `Sim` class. Let's visualise these functions, as well as the constrained objective formed by applying a mask to the objective over regions where the constraint function crosses the threshold.

# %%
util.plot_objective_and_constraints(search_space, Sim)
plt.show()

# %% [markdown]
# We'll make an observer that outputs the objective and constraint data, labelling each as shown.

# %%
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

def observer(query_points):
    return {
        OBJECTIVE: trieste.data.Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: trieste.data.Dataset(query_points, Sim.constraint(query_points))
    }

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
initial_data = observer(search_space.sample(5))

# %% [markdown]
# ... and visualise those points on the constrained objective.

# %%
util.plot_init_query_points(
    search_space,
    Sim,
    astuple(initial_data[OBJECTIVE]),
    astuple(initial_data[CONSTRAINT])
)
plt.show()

# %% [markdown]
# ## Modelling the two functions
#
# We'll model the objective and constraint data with their own Gaussian process regression models.

# %%
def create_bo_model(data):
    variance = tf.math.reduce_variance(initial_data[OBJECTIVE].observations)
    lengthscale = 1.0 * np.ones(2, dtype=default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    gpr = gpflow.models.GPR(astuple(data), kernel, noise_variance=1e-5)
    set_trainable(gpr.likelihood, False)
    return trieste.models.create_model(
        {
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {"options": dict(maxiter=100)},
        }
    )

models = {
    OBJECTIVE: create_bo_model(initial_data[OBJECTIVE]),
    CONSTRAINT: create_bo_model(initial_data[CONSTRAINT])
}

# %% [markdown]
# ## Define the acquisition process
#
# We can construct the _expected constrained improvement_ acquisition function defined in Gardner, 2014 [1], where they use the probability of feasibility wrt the constraint model.
#
# The previous criterion cannot be computed in closed form, so we'll use a Monte-Carlo estimate.
# %%
pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim.threshold)
eci = trieste.acquisition.ExpectedConstrainedImprovement(OBJECTIVE, pof.using(CONSTRAINT))
rule = trieste.acquisition.rule.EfficientGlobalOptimization(eci)

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the optimization loop

# %%
num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)

if result.error is not None: raise result.error

# %% [markdown]
# Finally, we visualise the resulting data. Orange dots show the new points queried during optimization. Notice the concentration of these points in regions near the local minima.

# %%
constraint_data = result.datasets[CONSTRAINT]
new_data = (
    constraint_data.query_points[-num_steps:], constraint_data.observations[-num_steps:]
)

util.plot_init_query_points(
    search_space,
    Sim,
    astuple(initial_data[OBJECTIVE]),
    astuple(initial_data[CONSTRAINT]),
    new_data
)
plt.show()

# %% [markdown]
# Now, we will show how to solve the same problem, but when point are queried in (synchronous) batches instead of one at a time.

# %%
class BatchExpectedConstrainedImprovement(AcquisitionFunctionBuilder):
    """
    Builder for the _expected constrained improvement_ acquisition function defined in
    Gardner, 2014, in a batch form, relying on Monte-Carlo.
    """

    def __init__(
        self,
        objective_tag: str,
        constraint_builder: AcquisitionFunctionBuilder,
        min_feasibility_probability: Union[float, tf.Tensor] = 0.5,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        tf.debugging.assert_scalar(min_feasibility_probability)

        if not 0 <= min_feasibility_probability <= 1:
            raise ValueError(
                f"Minimum feasibility probability must be between 0 and 1 inclusive,"
                f" got {min_feasibility_probability}"
            )

        self._objective_tag = objective_tag
        self._constraint_builder = constraint_builder
        self._min_feasibility_probability = min_feasibility_probability

    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: The expected constrained improvement acquisition function.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise ValueError: If the objective data is empty.
        """
        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        if len(objective_dataset) == 0:
            raise ValueError(
                "Expected improvement is defined with respect to existing points in the objective"
                " data, but the objective data is empty."
            )

        constraint_fn = self._constraint_builder.prepare_acquisition_function(datasets, models)
        pof = constraint_fn(objective_dataset.query_points)
        is_feasible = pof >= self._min_feasibility_probability

        if not tf.reduce_any(is_feasible):
            return constraint_fn

        mean, _ = objective_model.predict(objective_dataset.query_points)
        eta = tf.reduce_min(tf.boolean_mask(mean, is_feasible), axis=0)

        return lambda at: expected_improvement(objective_model, eta, at) * constraint_fn(at)



# %% [markdown]
# ## References
#
# ```
# [1] @inproceedings{gardner14,
#       title={Bayesian Optimization with Inequality Constraints},
#       author={
#         Jacob Gardner and Matt Kusner and Zhixiang and Kilian Weinberger and John Cunningham
#       },
#       booktitle={Proceedings of the 31st International Conference on Machine Learning},
#       year={2014},
#       volume={32},
#       number={2},
#       series={Proceedings of Machine Learning Research},
#       month={22--24 Jun},
#       publisher={PMLR},
#       url={http://proceedings.mlr.press/v32/gardner14.html},
#     }
# ```
