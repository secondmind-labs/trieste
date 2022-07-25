# %% [markdown]
# # An overview of Trieste types

# %% [markdown]
# Trieste is dedicated to Bayesian optimization, the process of finding the *optimal values of an expensive, black-box objective function by employing probabilistic models over observations*. This notebook explains how the different parts of this process are represented by different types in the code, and how these types can be extended.

# %% [markdown]
# ## Key types
#
# The following types represent the key concepts in Trieste. For a full listing of all the types in Trieste, see the API documentation.

# %% [markdown]
# ### `Observer`
#
# The `Observer` type definition represents the black box objective function. Observers are functions that accept query points and return observations. Observations are either a single objective value that we wish to optimize, or a dictionary of mutiple tagged values that must be combined somehow, for example an objective and an inequality constraint.

# %% [markdown]
# ### `Dataset`
#
# The `Dataset` container class represents the query points and observations from a single observer. Observers with multiple observations are represented by a dictionary of multiple tagged `Dataset`s.

# %% [markdown]
# ### `ProbabilisticModel`
#
# The `ProbabilisticModel` protocol represents any probabilistic model used to model observations. Like for `Dataset`, there may be multiple tagged models stored in a dictionary.
#
# At it simplest, `ProbabilisticModel` is anything that implements a `predict` and `sample` method. However, many algorithms in Trieste depend on models with additional features, which are represented by the various subclasses of `ProbabilisticModel`. The standard Bayesian optimizer uses `TrainableProbabilisticModel` models, which also implement `update` and `optimize` methods. Specific acuqisition functions may require other features, represented by classes like `SupportsPredictJoint` and `SupportsGetObservationNoise`. Since these are defined as protocols, it is possible to define and depend on the intersections of different model types (e.g. only support models that are both `SupportsPredictJoint` and `SupportsGetObservationNoise`).

# %% [markdown]
# ### `SearchSpace`
#
# The `SearchSpace` base class represents the domain over which the objective function is to be optimized. Spaces can currently be either continuous `Box` spaces, discrete `DiscreteSearchSpace` spaces, or a `TaggedProductSearchSpace` product of multipe spaces.

# %% [markdown]
# ### `AcquisitionRule`
#
# The `AcquisitionRule` base class represents a routine for selecting new query points during a Bayesian optimization loop (via an `acquire` method). It is generic on three types:
#
# * **ResultType**: the output of the rule, typically this is just tensors representing the query points. However, it can also be functions that accept some *acquisition state* and return the query points with a new state.
# * **SearchSpaceType**: the type of the search space; any optimizer that the rule uses must support this.
# * **ProbabilisticModelType**: the type of the models; any acquisition functions or samplers that the rule uses must support this.
#
# Exampes of rules include:
#
# 1. `EfficientGlobalOptimization` (EGO) is the most commonly used rule, and uses acquisition functions and optimizers to select new query points.
# 1. `AsynchronousOptimization` is similar to EGO but uses acquisition state to keep track of points who observations are still pending.
# 1. `DiscreteThompsonSampling` uses Thompson samplers rather than acquisition functions to select new query points.
#

# %% [markdown]
# ### `AcquisitionFunction` and `AcquisitionFunctionBuilder`
#
# The `AcquisitionFunction` type definition represents any acquisition function: that is, a function that maps a set of query points to a single value that describes how useful it would be evaluate all these points together.
#
# The `AcquisitionFunctionBuilder` base class, meanwhile, represents something that builds and updates acquisition functions. *Much of Trieste's codebase involves defining these builders.* At the start of the Bayesian optimization, the builder's `prepare_acquisition_function` method is called by the acquisition rule to create an acquisition function from the current observations and probabilistic models. For efficiency, most builders also define an `update_acquisition_function` method for updating the function using the updated observations and models. (The ones that don't instead generate a new acquisition function when necessary.)
#
# Acquisition functions that support only one probabilistic model are more easily defined using the `SingleModelAcquisitionBuilder` convenience class.

# %% [markdown]
# ### `AcquisitionOptimizer`
#
# The `AcquisitionOptimizer` type definition represents an optimizer function that maximizes an acquisition function over a search space.

# %% [markdown]
# ### `BayesianOptimizer` and `AskTellOptimizer`
#
# The `BayesianOptimizer` and `AskTellOptimizer` classes are the two Bayesian optimization interfaces provided by Trieste.
#
# `BayesianOptimizer` exposes an `optimize` method for running a Bayesian optimization loop with given initial datasets and models, and a given number of steps (and an optional early stop callback).
#
# `AskTellOptimizer` provides greater control over the loop, by providing separate `ask` and `tell` steps for suggesting query points and updating the models with new observations.

# %% [markdown]
# ## Extending the key types

# %%
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from trieste.types import TensorType

# %% [markdown]
# This section explains how to define new observers, model types and acqusition functions.

# %% [markdown]
# ### New observers
#
# Defining an observer with a single observation is as simple as defining a function that returns that observation:

# %%
from trieste.objectives.utils import mk_observer


def simple_quadratic(x: TensorType) -> TensorType:
    "A trivial quadratic function over :math:`[0, 1]^2`."
    return -tf.math.reduce_sum(x, axis=-1, keepdims=True) ** 2


observer = mk_observer(simple_quadratic)
observer(tf.constant([[0, 1], [1, 1]], dtype=tf.float64))

# %% [markdown]
# A multi-observation observer can be constructed from multiple functions:

# %%
from trieste.objectives.utils import mk_multi_observer


def simple_constraint(x: TensorType) -> TensorType:
    "A trivial constraints over :math:`[0, 1]^2`."
    return tf.math.reduce_min(x, axis=-1, keepdims=True)


multiobserver = mk_multi_observer(
    OBJECTIVE=simple_quadratic, CONSTRAINT=simple_constraint
)
multiobserver(tf.constant([[0, 1], [1, 1]], dtype=tf.float64))

# %% [markdown]
# ### New probabilistic model types
#
# Defining a new probabilistic model type simply involves writing a class that implements all the relevant methods (at the very least `predict` and `sample`). For clarity, it is best to also explicitly inherit from the supported feature protocols.

# %%
from trieste.models.interfaces import (
    TrainableProbabilisticModel,
    HasReparamSampler,
    HasTrajectorySampler,
    ReparametrizationSampler,
)


class GizmoModel(
    TrainableProbabilisticModel, HasReparamSampler, HasTrajectorySampler
):
    "A pretend trainable model type with reparametrization and trajectory samplers."

    def predict(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        ...

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[GizmoModel]:
        ...

    ...


# %% [markdown]
# If the new model type has an additional feature on which you'd like to depend, e.g. in a new acquisition function, then you can define that feature as a protocol. Marking it runtime_checkable will alow you to check for the feature at runtime too.

# %%
from trieste.models.interfaces import ProbabilisticModel
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasGizmo(ProbabilisticModel, Protocol):
    "A probabilistic model that has a 'gizmo' method."

    def gizmo(self) -> int:
        "A 'gizmo' method."
        raise NotImplementedError


# %% [markdown]
# If your acquisition function depends on a combination of features, then you can define an intersection protocol and use it when defining the acquisition function:

# %%
@runtime_checkable
class HasGizmoTrajectoryAndReparamSamplers(
    HasGizmo, HasReparamSampler, HasTrajectorySampler, Protocol
):
    """A model that supports both gizmo, reparam_sampler and trajectory_sampler."""

    pass


# %% [markdown]
# ### New acquisition function builders
#
# To define a new acquisition function builder, you simply need to define a class with a `prepare_acquisition_function` method that returns an `AcquisitionFunction`. If the acquisition function depends on just one model/dataset (as is often the case) then you can define it as a `SingleModelAcquisitionBuilder`; if it depends on more than one (e.g. both an objective and a constraint) then you must define it as a `ModelAcquisitionBuilder` instead. You can also specify, in brackets, the type of probabilistic models that the acquisition function supports (e.g. a `SingleModelAcquisitionBuilder[HasReparamSampler]` only supports models with a reparatemtrization sampler). This allows the type checker to warn you if you try to use it with an incompatible model type.

# %%
from trieste.acquisition import (
    AcquisitionFunction,
    SingleModelAcquisitionBuilder,
)
from trieste.data import Dataset


class ProbabilityOfValidity(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    def prepare_acquisition_function(
        self, model: ProbabilisticModel, dataset: Optional[Dataset] = None
    ) -> AcquisitionFunction:
        def acquisition(at: TensorType) -> TensorType:
            mean, _ = model.predict_y(tf.squeeze(at, -2))
            return mean

        return acquisition


# %% [markdown]
# For efficiency, it usually makes sense to compile the generated acquisition function into a TensorFlow graph using `tf.function`. Furthermore, to avoid generating (and compiling) a new acquisition function on each Bayesian optimization loop, you can define an `update_acquisition_function` method that can instead update the previously generated acquisition function using the new models and data. This may involve updating the acquisition function's internal state (which you should store in `tf.Variable`s), though if the function has no internal state then it is suficient to simply return the old function unchanged.

# %%
class ProbabilityOfValidityEfficient(
    SingleModelAcquisitionBuilder[ProbabilisticModel]
):
    def prepare_acquisition_function(
        self, model: ProbabilisticModel, dataset: Optional[Dataset] = None
    ) -> AcquisitionFunction:
        @tf.function
        def acquisition(at: TensorType) -> TensorType:
            mean, _ = model.predict_y(tf.squeeze(at, -2))
            return mean

        return acquisition

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        return function  # no need to update anything


# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
