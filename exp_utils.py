import tensorflow as tf

import gpflow
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseGaussianProcessRegression,
    build_gpr,
    build_sgpr
)
from trieste.models.gpflux import (
    DeepGaussianProcess,
    build_vanilla_deep_gp,
)
from trieste.acquisition.rule import (
    EfficientGlobalOptimization,
)
from trieste.acquisition.function import (
    GreedyContinuousThompsonSampling,
    ParallelContinuousThompsonSampling
)
from trieste.models.optimizer import KerasOptimizer, Optimizer
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.space import SearchSpace
from trieste.data import Dataset


def build_sgpr_model(
    data: Dataset,
    search_space: SearchSpace,
    num_inducing: int = 100,
    learn_noise: bool = False,
    epochs: int = 400
):
    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = 1e-5

    sgpr = build_sgpr(data, search_space, likelihood_variance=noise_variance,
                    trainable_likelihood=learn_noise,
                      num_inducing_points=num_inducing,
                      trainable_inducing_points=True)

    acquistion_function = GreedyContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(acquistion_function,
                                                   num_query_points=1)

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True) #, minimize_args=dict(options=dict(disp=True)))

    return SparseGaussianProcessRegression(sgpr, optimizer), acquisition_rule


def build_gp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_inducing: int = 100,
        learn_noise: bool = False,
        epochs: int = 400
):
    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = 1e-5

    gpr = build_gpr(data, search_space, likelihood_variance=noise_variance,
                    trainable_likelihood=learn_noise)

    acquistion_function = GreedyContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(acquistion_function,
                                                   num_query_points=1)

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True) #, minimize_args=dict(options=dict(disp=True)))

    return GaussianProcessRegression(gpr, optimizer), acquisition_rule


def build_vanilla_dgp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_layers: int = 2,
        num_inducing: int = 100,
        learn_noise: bool = False,
        epochs: int = 400
):
    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = 1e-5

    dgp = build_vanilla_deep_gp(data, search_space, num_layers, num_inducing,
                                likelihood_variance=noise_variance, trainable_likelihood=learn_noise)

    acquisition_function = GreedyContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(acquisition_function,
                                                   num_query_points=1)
                                                   # optimizer=generate_continuous_optimizer(1000))

    batch_size = 1000

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    keras_optimizer = tf.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": True,
        "callbacks": [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    }
    optimizer = KerasOptimizer(keras_optimizer, fit_args)

    return (
        DeepGaussianProcess(dgp, optimizer, continuous_optimisation=False),
        acquisition_rule
    )


def build_svgp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_inducing: int = 100,
        learn_noise: bool = False,
        epochs: int = 400
):
    return build_vanilla_dgp_model(data, search_space, num_layers=1, num_inducing=num_inducing,
                                   learn_noise=learn_noise, epochs=epochs)


def normalize(x, mean=None, std=None):
    if mean is None:
        mean = tf.math.reduce_mean(x, 0, True)
    if std is None:
        std = tf.math.sqrt(tf.math.reduce_variance(x, 0, True))
    return (x - mean) / std, mean, std
