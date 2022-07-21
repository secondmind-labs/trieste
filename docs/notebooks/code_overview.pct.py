# %% [markdown]
# # Trieste code overview

# %% [markdown]
# Trieste is dedicated to Bayesian optimization, the process of finding the *optimal values of an expensive, black-box objective function by employing probabilistic models over observations*. This notebook explains how the different parts of this process are represented by different classes, and how these classes can be extended.

# %% [markdown]
# ## Key classes
#
# The following classes represent the key concepts in Trieste. For a full listing of the classes in Trieste, see the API documentation.

# %% [markdown]
# ### `Observer`
#
# The `Observer` type definition represents the black box objective function. Observers are functions that accept query points and return observations. Observations are either a single objective vaue that we wish to optimize, or mutiple tagged values that must be combined somehow, for example an objective and an inequality constraint.

# %% [markdown]
# ### `Dataset`
#
# The `Dataset` container class represents the query points and observations from a single observer. Observers with multiple observations are represented by a dictionary of multiple, tagged `Dataset`s.

# %% [markdown]
# ### `ProbabilisticModel`
#
# The `ProbabilisticModel` protocol represents any probabilistic model used to model observations. Like for `Dataset`, there may be multiple, tagged models stored in a dictionary.
#
# At it simplest, `ProbabilisticModel` is anything that implements a `predict` and `sample` method. However, many algorithms in Trieste depend on models with additional features, which are represented by the various subclasses of `ProbabilisticModel`. The standard Bayesian optimizer uses `TrainableProbabilisticModel` models, which also implement `update` and `optimize` methods. Specific acuqisition functions may require other features, represented by classes like `SupportsPredictJoint` and `SupportsGetObservationNoise`. Since these are defined as protocols, it is possible to define and depend on the intersections of different model types (e.g. only support models that are both `SupportsPredictJoint` and `SupportsGetObservationNoise`).

# %% [markdown]
# ### `SearchSpace`
#
# The `SearchSpace` base class represents the domain over which objective function is to be optimized. Spaces can currently be either continuous `Box` spaces, discrete `DiscreteSearchSpace` spaces or a `TaggedProductSearchSpace` product of multipe spaces.

# %% [markdown]
# ### `AcquisitionRule`
#
# The `AcquisitionRule` base class represents a routine for generating new query points (via an `acquire` method). It is generic on three types:
#
# * **ResultType**: the output of the rule, typically this is just tensors representing the query points. However, it can also be functions that accept some *acquisition state* and return the query points with a new state.
# * **SearchSpaceType**: the type of the search space; any optimizer that the rule uses must support this.
# * **ProbabilisticModelType**: the type of the models; any acquisition functions or samplers that the rule uses must support this.
#
# Exampes of rules include:
#
# 1. `EfficientGlobalOptimization` (EGO) is the most commonly used rule, and uses acquisition functions and optimizers.
# 1. `AsynchronousOptimization` is similar to EGO but uses acquisition state to keep track of pending points.
# 1. `DiscreteThompsonSampling` uses Thompson samplers rather than acquisition functions.
#

# %% [markdown]
# ### `AcquisitionFunction` and `AcquisitionFunctionBuilder`
#
# The `AcquisitionFunction` type definition represents any acquisition function: that is, a function that maps a set of query points to a single value that describes how useful it would be evaluate all these points together.
#
# The `AcquisitionFunctionBuilder` base class, meanwhile, represents something that builds and updates acquisition functions. *Much of Trieste's codebase involves defining these builders.* At the start of the Bayesian optimization, the builder's `prepare_acquisition_function` method is called by the rule to create an acquisition function from the current observations and probabilistic models. For efficiency, most builders also defing an `update_acquisition_function` method for updating the function using the updated observations and models. (The ones that don't instead generate a new acquisition function when necessary.)
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
# `AskTellOptimizer` provides greater control over the loop, by providing separate `ask` and `tell` steps for suggesting points and updating the models with new observ

# %% [markdown]
# ## Extending the key classes

# %%
import tensorflow as tf
from trieste.types import TensorType


# %% [markdown]
# This section explains how to define new observers, acqusition functions and model types.

# %% [markdown]
# ### Observers
#
# Defining an observer is as simple as defining a function that returns observations:

# %%
def simple_quadratic(x: TensorType) -> TensorType:
    "A trivial quadratic function over :math:`[0, 1]^2`."
    return -tf.math.reduce_sum(x, axis=-1, keepdims=True) ** 2

# %% [markdown]
# A multi-observation observer returns instead a dictionary of observations:

# %%

# %% [markdown]
# ### Acquisition function builders

# %% [markdown]
# ### Probabilistic model types

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
