import numpy as np

import lunar_lander
from turbo_test import demo_heuristic_lander
import tensorflow as tf
import tensorflow_probability as tfp
import trieste

import timeit

# this space is created by going approximately +-0.2 around parameter values, but not going below 0
# see for original values https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
# original values are
# 0.5 1 0.4 0.55 0.5 1 0.5 0.5 0 0.5 0.05 0.05
# and for parameter definition https://github.com/uber-research/TuRBO
search_space = trieste.space.Box(
    [0.3, 0.8, 0.2, 0.35, 0.3, 0.8, 0.3, 0.3, 0.0, 0.3, 0.0,  0.0],
    [0.7, 1.2, 0.6, 0.75, 0.7, 1.2, 0.7, 0.7, 0.2, 0.7, 0.25, 0.25]
)

# lander landed, minimize fuel
FUEL = "FUEL"
# minimize failures, when the lander crashes or times out
FAILURE = "FAILURE"

create_empty_dataset = lambda : trieste.data.Dataset(
                                    tf.zeros((0, search_space.dimension), tf.float64),
                                    tf.zeros((0, 1), tf.float64)
                                )

def lander_observer(x):
    all_datasets = {
        FUEL: create_empty_dataset(),
        FAILURE: create_empty_dataset()
    }

    def add_data(dataset_tag, x, y):
        new_dataset = trieste.data.Dataset(np.atleast_2d(x), np.atleast_2d(y))
        all_datasets[dataset_tag] += new_dataset

    for w in x.numpy():
        result = demo_heuristic_lander(lunar_lander.LunarLander(), w)
        # that's different from constrained optimization
        # because now we want to minimize failure
        # and not penalize by it
        # thus we flip the 1 and 0 values in this dataset
        if result.timeout or result.has_crashed:
            add_data(FAILURE, w, 1.0)
            continue
        else:
            add_data(FAILURE, w, 0.0)

        normalized_fuel = np.float64(result.total_fuel / 100.0)
        add_data(FUEL, w, normalized_fuel)
    
    return all_datasets


num_initial_points = 1
initial_query_points = search_space.sample(1)
initial_data = lander_observer(initial_query_points)

# collect points until we have at least one in each dataset
while any(len(initial_data[tag]) < search_space.dimension for tag in initial_data):
    initial_query_points = search_space.sample(1)
    new_initial_data = lander_observer(initial_query_points)
    for tag in initial_data:
        initial_data[tag] = initial_data[tag] + new_initial_data[tag]
    num_initial_points += 1


print(len(initial_data[FUEL]))
print(len(initial_data[FAILURE]))

import gpflow


def create_regression_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance, lengthscales=[0.2]*int(search_space.dimension))
    scale = tf.constant(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.constant(-2.0, dtype=tf.float64), scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel)
    return gpr


def create_classification_model(data):
    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=[0.2]*int(search_space.dimension)
    )
    likelihood = gpflow.likelihoods.Bernoulli()
    vgp = gpflow.models.VGP(data.astuple(), kernel, likelihood)
    return vgp

from trieste.models.gpflow import GPflowModelConfig

classification_model_config_args = {
    "model_args": {"use_natgrads": True},
    "optimizer": tf.optimizers.Adam(1e-3),
    "optimizer_args": {"max_iter": 50},
}
models = {
    FUEL: GPflowModelConfig(**{
        "model": create_regression_model(initial_data[FUEL]),
        "optimizer": gpflow.optimizers.Scipy(),
    }),
    FAILURE: GPflowModelConfig(
        create_classification_model(initial_data[FAILURE]),
        **classification_model_config_args
    )
}


class SpecialModelStack(trieste.models.ModelStack):
    """ Special treatment of predict_joint used in sampler
    """
    def __init__(self, models_dict):
        super().__init__((models_dict[FUEL], 1), (models_dict[FAILURE], 1))
        self._models_dict = models_dict

    def predict(self, query_points):
        fuel_mean, fuel_var = self._models_dict[FUEL].predict(query_points)
        failure_mean, failure_var = self._models_dict[FAILURE].predict_y(query_points)
        return tf.concat([fuel_mean, failure_mean], axis=-1), tf.concat([fuel_var, failure_var], axis=-1)


class SpecialBatchMonteCarloExpectedHypervolumeImprovement(trieste.acquisition.function.AcquisitionFunctionBuilder):
    """ The one in trieste is single model, and we need to pass two models and two datasets
    """

    def __init__(self, sample_size: int, *, jitter: float = trieste.utils.misc.DEFAULTS.JITTER):
        """
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return (
            f"SpecialBatchMonteCarloExpectedHypervolumeImprovement({self._sample_size!r},"
            f" jitter={self._jitter!r})"
        )

    def prepare_acquisition_function(
        self,
        models,
        datasets,
    ):
        # failure dataset will have all points
        # while fuel only successful ones
        query_points = datasets[FAILURE].query_points
        
        # [0] is because we only need mean and not variance
        means = tf.concat([models[FUEL].predict(query_points)[0], models[FAILURE].predict_y(query_points)[0]], axis=-1)
        _pf = trieste.acquisition.multi_objective.pareto.Pareto(means)
        _reference_pt = trieste.acquisition.multi_objective.pareto.get_reference_point(_pf.front)
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = trieste.acquisition.multi_objective.partition.prepare_default_non_dominated_partition_bounds(_reference_pt, _pf.front)

        sampler = trieste.acquisition.sampler.IndependentReparametrizationSampler(self._sample_size, SpecialModelStack(models))

        return trieste.acquisition.function.batch_ehvi(sampler, self._jitter, _partition_bounds)


from trieste.acquisition.rule import EfficientGlobalOptimization

BATCH_SIZE = 1
ITERATIONS = 2

mc_ehvi = SpecialBatchMonteCarloExpectedHypervolumeImprovement(sample_size=5)
rule = EfficientGlobalOptimization(mc_ehvi, num_query_points=BATCH_SIZE)


bo = trieste.bayesian_optimizer.BayesianOptimizer(lander_observer, search_space)
start = timeit.default_timer()
result = bo.optimize(ITERATIONS, initial_data, models, rule).final_result.unwrap()
stop = timeit.default_timer()