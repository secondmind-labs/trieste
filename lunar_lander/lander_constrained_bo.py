import numpy as np

import lunar_lander
from turbo_test import demo_heuristic_lander
import tensorflow as tf
import trieste

# this space is created by doing +-0.1 around parameter values
# set in https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
# search_space = trieste.space.Box(
#     [0.4, 0.9, 0.3, 0.5, 0.4, 0.9, 0.4, 0.4, 0.0, 0.5, 0.0, 0.0],
#     [0.6, 1.1, 0.5, 0.6, 0.6, 1.1, 0.6, 0.6, 0.1, 0.6, 0.1, 0.1]
# )

search_space = trieste.space.Box([0.0], [1.5]) ** 12


OBJECTIVE = "OBJECTIVE"
# lander crashed
CRASH = "CRASH"
# didn't finish in predefined number of steps
TIMEOUT = "TIMEOUT"
# lander landed fine, but outside the helipad
OUTSIDE = "OUTSIDE"

create_empty_dataset = lambda : trieste.data.Dataset(tf.zeros((0, search_space.dimension), tf.float64), tf.zeros((0, 1), tf.float64))

def lander_observer(x):
    all_datasets = {
        OBJECTIVE: create_empty_dataset(),
        CRASH: create_empty_dataset(),
        TIMEOUT: create_empty_dataset(),
        OUTSIDE: create_empty_dataset()
    }

    def add_data(dataset_tag, x, y):
        new_dataset = trieste.data.Dataset(np.atleast_2d(x), np.atleast_2d(y))
        all_datasets[dataset_tag] += new_dataset

    for w in x.numpy():
        result = demo_heuristic_lander(lunar_lander.LunarLander(), w)
        if result.timeout:
            add_data(TIMEOUT, w, 0.0)
            continue
        else:
            add_data(TIMEOUT, w, 1.0)

        if result.has_crashed:
            add_data(CRASH, w, 0.0)
            continue
        else:
            add_data(CRASH, w, 1.0)

        if not result.is_in_helipad:
            add_data(OUTSIDE, w, 0.0)
            continue
        else:
            add_data(OUTSIDE, w, 1.0)

        # all failure modes are done, means we landed successfully
        add_data(OBJECTIVE, w, result.total_fuel)
    
    return all_datasets


num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = lander_observer(initial_query_points)

print("---------------- Initial data generated ------------------")

import gpflow


def create_regression_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.1]*int(search_space.dimension))
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return gpr


def create_classification_model(data):
    kernel = gpflow.kernels.SquaredExponential(
        variance=100.0, lengthscales=[0.1]*int(search_space.dimension)
    )
    likelihood = gpflow.likelihoods.Bernoulli()
    vgp = gpflow.models.VGP(data.astuple(), kernel, likelihood)
    gpflow.set_trainable(vgp.kernel.variance, False)
    return vgp

from trieste.models.gpflow import GPflowModelConfig

classification_model_config_args = {
    "model_args": {"use_natgrads": True},
    "optimizer": tf.optimizers.Adam(1e-3),
    "optimizer_args": {"max_iter": 50},
}
models = {
    OBJECTIVE: GPflowModelConfig(**{
        "model": create_regression_model(initial_data[OBJECTIVE]),
        "optimizer": gpflow.optimizers.Scipy(),
    }),
    CRASH: GPflowModelConfig(
        create_classification_model(initial_data[CRASH]),
        **classification_model_config_args
    ),
    TIMEOUT: GPflowModelConfig(
        create_classification_model(initial_data[TIMEOUT]),
        **classification_model_config_args
    ),
    OUTSIDE: GPflowModelConfig(
        create_classification_model(initial_data[OUTSIDE]),
        **classification_model_config_args
    ),
}


from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import (
    SingleModelAcquisitionBuilder, ExpectedImprovement, Product
)

class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(self, model, dataset=None):
        def acquisition(at):
            mean, _ = model.predict_y(tf.squeeze(at, -2))
            return mean
        return acquisition

acq_fn = Product(
    ExpectedImprovement().using(OBJECTIVE),
    ProbabilityOfValidity().using(CRASH),
    ProbabilityOfValidity().using(TIMEOUT),
    ProbabilityOfValidity().using(OUTSIDE),
)
rule = EfficientGlobalOptimization(acq_fn)

bo = trieste.bayesian_optimizer.BayesianOptimizer(lander_observer, search_space)

result = bo.optimize(100, initial_data, models, rule).final_result.unwrap()

print("phew")