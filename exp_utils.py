import gpflow
import tensorflow as tf
import bayesfunc as bf

from typing import Optional
import trieste.models.gpflux.models
from trieste.models.gpflux import (
    GPfluxModelConfig,
    build_vanilla_deep_gp,
    build_latent_variable_dgp_model,
    build_gi_deep_gp,
    build_dkp_model
)
from trieste.models.bayesfunc import (
    BayesFuncModel,
    build_sqexp_deep_inv_wishart
)
from trieste.acquisition.rule import (
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
)
from trieste.acquisition.models import (
    DeepGaussianProcessSampler,
    GlobalInducingDeepGaussianProcessSampler,
    DeepKernelProcessSampler
)
from trieste.acquisition.function import (
    MonteCarloExpectedImprovement,
    MonteCarloAugmentedExpectedImprovement,
    ExpectedImprovement,
    AugmentedExpectedImprovement,
    NegativeModelTrajectory
)
from trieste.acquisition.optimizer import generate_continuous_optimizer
from gpflow.utilities import set_trainable
import tensorflow_probability as tfp
from trieste.models.gpflow import GPflowModelConfig
from trieste.data import TensorType, Dataset
from trieste.space import Box
from scipy.stats import norm
import numpy as np
import math
from gpflow.utilities import positive, print_summary
from gpflow.base import Parameter
import torch as t


def build_deep_kernel_process_model(data, num_layers=2, num_inducing=100, learn_noise: bool = False,
                    search_space: Optional[Box] = None):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_dkp_model(data.query_points, num_layers=num_layers, num_inducing=num_inducing,
                           last_layer_variance=variance.numpy(), num_train_samples=1,
                           search_space=search_space)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant(
        tf.reduce_mean(data.observations))
    if learn_noise:
        dgp.likelihood_layer.variance.assign(1e-3)
        MCEI = MonteCarloAugmentedExpectedImprovement(DeepKernelProcessSampler,
                                                      10)
        acquisition_rule = EfficientGlobalOptimization(MCEI)
    else:
        dgp.likelihood_layer.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)
        MCEI = MonteCarloExpectedImprovement(DeepKernelProcessSampler, 10)
        acquisition_rule = EfficientGlobalOptimization(MCEI)

    acquisition_function = NegativeModelTrajectory()
    acquisition_rule = EfficientGlobalOptimization(acquisition_function, optimizer=generate_continuous_optimizer(1000))

    optimizer = tf.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.5)

    return (
        GPfluxModelConfig(**{
            "model": dgp,
            "optimizer": optimizer,
        }),
        acquisition_rule
    )


def build_vanilla_dgp_model(data, num_layers=2, num_inducing=100, learn_noise: bool = False,
                            search_space: Optional[Box] = None):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_vanilla_deep_gp(data.query_points, num_layers=num_layers, num_inducing=num_inducing,
                                search_space=search_space)
    dgp.f_layers[-1].kernel.kernel.variance.assign(variance)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant(tf.reduce_mean(data.observations))
    if learn_noise:
        dgp.likelihood_layer.likelihood.variance.assign(1e-3)
        # MCEI = MonteCarloAugmentedExpectedImprovement(DeepGaussianProcessSampler, 10)
        acquisition_function = NegativeModelTrajectory()
        acquisition_rule = EfficientGlobalOptimization(acquisition_function, optimizer=generate_continuous_optimizer(1000))
    else:
        dgp.likelihood_layer.likelihood.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)
        # MCEI = MonteCarloExpectedImprovement(DeepGaussianProcessSampler, 10)
        acquisition_function = NegativeModelTrajectory()
        acquisition_rule = EfficientGlobalOptimization(acquisition_function, optimizer=generate_continuous_optimizer(1000))

    epochs = 400
    batch_size = 1000

    optimizer = tf.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": False,
    }

    return (
        GPfluxModelConfig(**{
            "model": dgp,
            "model_args": {
                "fit_args": fit_args,
            },
            "optimizer": optimizer,
        }),
        acquisition_rule
    )


def build_gi_dgp_model(data, num_layers=2, num_inducing=100, learn_noise: bool = False,
                       search_space: Optional[Box] = None):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_gi_deep_gp(data.query_points, num_layers=num_layers, num_inducing=num_inducing,
                           last_layer_variance=variance.numpy(), num_train_samples=1,
                           search_space=search_space)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant(tf.reduce_mean(data.observations))
    if learn_noise:
        dgp.likelihood_layer.variance.assign(1e-3)
        # MCEI = MonteCarloAugmentedExpectedImprovement(GlobalInducingDeepGaussianProcessSampler, 10)
        # acquisition_rule = EfficientGlobalOptimization(MCEI)
    else:
        dgp.likelihood_layer.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)
        # MCEI = MonteCarloExpectedImprovement(GlobalInducingDeepGaussianProcessSampler, 10)
        # acquisition_rule = EfficientGlobalOptimization(MCEI)

    acquisition_function = NegativeModelTrajectory()
    acquisition_rule = EfficientGlobalOptimization(acquisition_function, optimizer=generate_continuous_optimizer(1000))

    epochs = 400
    batch_size = 1000

    optimizer = tf.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
    }

    return (
        GPfluxModelConfig(**{
            "model": dgp,
            "model_args": {
                "fit_args": fit_args,
            },
            "optimizer": optimizer,
        }),
        acquisition_rule
    )


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
    gpflow.config.set_default_jitter(1e-5)
    variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2]*dim)
    variance_prior_loc = variance.numpy()
    variance_prior_scale = 1.0
    lengthscales_prior_scale = 0.5
    kernel.lengthscales.assign(0.3*(search_space.upper - search_space.lower) * np.sqrt(dim))
    kernel.variance.prior = tfp.distributions.LogNormal(
        loc=np.float64(variance_prior_loc), scale=np.float64(variance_prior_scale)
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        loc=np.float64(np.log(kernel.lengthscales.numpy())), scale=np.float64(lengthscales_prior_scale)
    )
    if learn_noise:
        gpr = gpflow.models.GPR(data.astuple(), kernel,
                                mean_function=gpflow.mean_functions.Constant(tf.reduce_mean(data.observations)),
                                noise_variance=1e-3)
        EI = AugmentedExpectedImprovement()
        acquisition_rule = EfficientGlobalOptimization(EI)
    else:
        gpr = gpflow.models.GPR(data.astuple(), kernel,
                                mean_function=gpflow.mean_functions.Constant(tf.reduce_mean(data.observations)),
                                noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)
        EI = ExpectedImprovement()
        acquisition_rule = EfficientGlobalOptimization(EI)

    return (
        GPflowModelConfig(**{
            "model": gpr,
            "model_args": {
                "num_kernel_samples": 100,
            },
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
        }),
        acquisition_rule
    )


def test_ll_dkp(
    data: Dataset,
    model: trieste.models.gpflux.models.DeepKernelProcess,
    num_samples: int = 100
) -> TensorType:
    model.num_test_samples = num_samples
    out_samples = model.model.call(data.query_points)
    l = out_samples.log_prob(data.observations)
    ind_ll = tf.reduce_logsumexp(l, axis=0) - math.log(num_samples)
    return tf.reduce_mean(ind_ll, axis=0).numpy()


# def test_ll_dkp(
#     data: Dataset,
#     model: trieste.models.bayesfunc.models.BayesFuncModel,
#     num_samples: int = 100
# ):
#     query_points = data.query_points
#     observations = data.observations
#     if isinstance(query_points, tf.Tensor):
#         query_points = query_points.numpy()
#     if isinstance(observations, tf.Tensor):
#         observations = observations.numpy()
#
#     with t.no_grad():
#         dkp = model.model
#         X = t.from_numpy(query_points).to(dtype=t.float64)
#         y = t.from_numpy(observations).to(dtype=t.float64).reshape(-1, 1)
#
#         X = X.expand(num_samples, *X.shape)
#         Py = dkp(X)
#         ind_ll = Py.log_prob(y)
#
#         test_ll = (t.logsumexp(ind_ll, 0) - math.log(num_samples)).mean(0)
#     return test_ll.numpy()


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
