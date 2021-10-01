import gpflow
import tensorflow as tf

from typing import Optional
import trieste.models.gpflux.models
from trieste.models.gpflux import (
    GPfluxModelConfig,
    build_vanilla_deep_gp,
    build_latent_variable_dgp_model,
    build_gi_deep_gp,
)
from gpflow.utilities import set_trainable
import tensorflow_probability as tfp
from trieste.models.gpflow import GPflowModelConfig
from trieste.data import TensorType, Dataset
from trieste.space import Box
from scipy.stats import norm
import numpy as np
import math


def build_vanilla_dgp_model(data, num_layers=2, num_inducing=200, learn_noise: bool = False,
                            search_space: Optional[Box] = None):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_vanilla_deep_gp(data.query_points, num_layers=num_layers, num_inducing=num_inducing,
                                search_space=search_space)
    dgp.f_layers[-1].kernel.kernel.variance.assign(variance)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant()
    if learn_noise:
        dgp.likelihood_layer.likelihood.variance.assign(1e-3)
    else:
        dgp.likelihood_layer.likelihood.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)

    epochs = 200
    batch_size = 1000

    optimizer = tf.optimizers.Adam(0.005)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": False,
    }

    return GPfluxModelConfig(**{
        "model": dgp,
        "model_args": {
            "fit_args": fit_args,
        },
        "optimizer": optimizer,
    })


def build_gi_dgp_model(data, num_layers=2, num_inducing=200, learn_noise: bool = False,
                       search_space: Optional[Box] = None):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_gi_deep_gp(data.query_points, num_layers=num_layers, num_inducing=num_inducing,
                           last_layer_variance=variance.numpy(), num_train_samples=1,
                           search_space=search_space)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant()
    if learn_noise:
        dgp.likelihood_layer.variance.assign(1e-3)
    else:
        dgp.likelihood_layer.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)

    epochs = 200
    batch_size = 1000

    optimizer = tf.optimizers.Adam(0.005)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
    }

    return GPfluxModelConfig(**{
        "model": dgp,
        "model_args": {
            "fit_args": fit_args,
        },
        "optimizer": optimizer,
    })


def build_lv_dgp_model(data, num_total_data, num_layers=2, num_inducing=200, latent_dim=1):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_latent_variable_dgp_model(data.query_points, num_total_data=num_total_data,
                                          num_layers=num_layers, num_inducing=num_inducing,
                                          latent_dim=latent_dim, prior_std=0.2)
    dgp.f_layers[-1].kernel.kernel.variance.assign(variance)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant()
    dgp.likelihood_layer.likelihood.variance.assign(1e-5)
    set_trainable(dgp.likelihood_layer, False)

    epochs = 500
    batch_size = 1000

    optimizer = tf.optimizers.Adam(0.005)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": False,
    }

    return GPfluxModelConfig(**{
        "model": dgp,
        "model_args": {
            "fit_args": fit_args,
        },
        "optimizer": optimizer,
    })


def build_gp_model(data, learn_noise: bool = False, search_space: Optional[Box] = None):
    gpflow.config.set_default_jitter(1e-4)
    print('jitter', gpflow.default_jitter())
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2]*data.query_points.shape[-1])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = None #tfp.distributions.LogNormal(tf.cast(-2.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = None #tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    if learn_noise:
        gpr = gpflow.models.GPR(data.astuple(), kernel, mean_function=gpflow.mean_functions.Constant(), noise_variance=1e-3)
    else:
        gpr = gpflow.models.GPR(data.astuple(), kernel, mean_function=gpflow.mean_functions.Constant(), noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)

    return GPflowModelConfig(**{
        "model": gpr,
        "model_args": {
            "num_kernel_samples": 100,
        },
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })


def test_ll_vanilla_dgp(
    data: Dataset,
    model: trieste.models.gpflux.models.DeepGaussianProcess,
    num_samples: int = 100
) -> TensorType:
    samples = []
    for _ in range(num_samples):
        out = model.model_gpflux.call(data.query_points)
        y_mean, y_var = out.y_mean, out.y_var
        l = norm.logpdf(data.observations.numpy(), loc=y_mean, scale=y_var**0.5)
        samples.append(l)
    samples = np.stack(samples)
    ind_ll = tf.reduce_logsumexp(samples, axis=0) - math.log(num_samples)
    return tf.reduce_mean(ind_ll, axis=0).numpy()


def test_ll_gi_dgp(
    data: Dataset,
    model: trieste.models.gpflux.models.GlobalInducingDeepGaussianProcess,
    num_samples: int = 100
) -> TensorType:
    model.num_test_samples = num_samples
    out_samples = model.model_gpflux.call(data.query_points)
    l = out_samples.log_prob(data.observations)
    ind_ll = tf.reduce_logsumexp(l, axis=0) - math.log(num_samples)
    return tf.reduce_mean(ind_ll, axis=0).numpy()


def test_ll_gp(
    data: Dataset,
    model: trieste.models.gpflow.GPflowPredictor
) -> TensorType:
    return tf.reduce_mean(model.model.predict_log_density(
        (data.query_points, data.observations)
    ), keepdims=True).numpy()
