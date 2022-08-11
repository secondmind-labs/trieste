import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.utilities import set_trainable, print_summary
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseGaussianProcessRegression,
    build_gpr
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
from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.space import SearchSpace
from trieste.data import Dataset
from trieste.types import TensorType
from scipy.cluster.vq import kmeans2
from trieste.models.gpflow.builders import _get_data_stats, _get_kernel, _get_mean_function, _set_gaussian_likelihood_variance
from trieste.models.gpflow.utils import SafeSGPR as SPGR

OPT_SPLIT_SIZE = 10_000


def build_sgpr_model(
    data: Dataset,
    search_space: SearchSpace,
    num_layers: int = 2,
    num_inducing: int = 100,
    learn_noise: bool = False,
    fix_ips: bool = False,
    epochs: int = 400,
    num_query_points: int = 1,
    noise_init: float = 1e-5
):
    if num_inducing > len(data.query_points):
        num_inducing = len(data.query_points)

    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = noise_init

    centroids, _ = kmeans2(data.query_points.numpy(), k=num_inducing, minit="points")

    emp_mean, emp_var, _ = _get_data_stats(data)
    kernel = _get_kernel(emp_var, search_space, True, True)
    mean = _get_mean_function(emp_mean)

    inducing_points = gpflow.inducing_variables.InducingPoints(centroids)

    sgpr = SPGR(data.astuple(), kernel, inducing_points, mean_function=mean)

    _set_gaussian_likelihood_variance(sgpr, emp_var, noise_variance)
    set_trainable(sgpr.likelihood, learn_noise)
    set_trainable(sgpr.inducing_variable, (not fix_ips))

    acquisition_function = ParallelContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(
        acquisition_function,
        num_query_points=num_query_points,
        optimizer=split_acquisition_function_calls(
            generate_continuous_optimizer(
                1000 * tf.shape(search_space.lower)[-1],
                5 * tf.shape(search_space.lower)[-1],
                ),
            split_size=OPT_SPLIT_SIZE
        )
    )

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True, minimize_args=dict(options=dict(maxiter=50)))

    def predict_mean(query_points: TensorType, model: SparseGaussianProcessRegression) -> TensorType:
        return model.predict(query_points)[0]

    return SparseGaussianProcessRegression(sgpr, optimizer), acquisition_rule, predict_mean


def build_gp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_layers: int = 2,
        num_inducing: int = 100,
        learn_noise: bool = False,
        fix_ips: bool = False,
        epochs: int = 400,
        num_query_points: int = 1,
        noise_init: float = 1e-5
):
    if learn_noise:
        noise_variance = 1e-1
    else:
        noise_variance = noise_init

    gpflow.config.set_default_jitter(1.e-5)
    gpflow.config.set_default_positive_minimum(1.e-5)

    gpr = build_gpr(data, search_space, likelihood_variance=noise_variance,
                    trainable_likelihood=learn_noise)

    acquisition_function = ParallelContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(
        acquisition_function,
        num_query_points=num_query_points,
        optimizer=split_acquisition_function_calls(
            generate_continuous_optimizer(
                1000 * tf.shape(search_space.lower)[-1],
                5 * tf.shape(search_space.lower)[-1],
                ),
            split_size=OPT_SPLIT_SIZE
        )
    )

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True) #, minimize_args=dict(options=dict(disp=True)))

    def predict_mean(query_points: TensorType, model: GaussianProcessRegression) -> TensorType:
        return model.predict(query_points)[0]

    return GaussianProcessRegression(gpr, optimizer), acquisition_rule, predict_mean


def build_vanilla_dgp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_layers: int = 2,
        num_inducing: int = 100,
        learn_noise: bool = False,
        fix_ips: bool = False,
        epochs: int = 400,
        num_query_points: int = 1,
        num_predict_samples: int = 100,
        noise_init: float = 1e-5
):
    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = noise_init

    dgp = build_vanilla_deep_gp(data, search_space, num_layers, num_inducing,
                                likelihood_variance=noise_variance, trainable_likelihood=learn_noise)

    if fix_ips:
        set_trainable(dgp.f_layers[0].inducing_variable, False)

    acquisition_function = ParallelContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(
        acquisition_function,
        num_query_points=num_query_points,
        optimizer=split_acquisition_function_calls(
            generate_continuous_optimizer(
                1000 * tf.shape(search_space.lower)[-1],
                5 * tf.shape(search_space.lower)[-1],
                ),
            split_size=OPT_SPLIT_SIZE
        )
    )

    batch_size = 1000

    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            'loss', factor=0.5, patience=5, min_lr=1e-8, verbose=0
    )

    keras_optimizer = tf.optimizers.Adam(0.1, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": True,
        "callbacks": [scheduler]
    }
    optimizer = KerasOptimizer(keras_optimizer, fit_args)

    def predict_mean(query_points: TensorType, model: DeepGaussianProcess) -> TensorType:
        samples = []
        for _ in range(num_predict_samples):
            samples.append(model.predict(query_points)[0])
        return tf.reduce_mean(tf.stack(samples), axis=0)

    return (
        DeepGaussianProcess(dgp, optimizer, continuous_optimisation=False),
        acquisition_rule,
        predict_mean
    )


def build_svgp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_inducing: int = 100,
        learn_noise: bool = False,
        fix_ips: bool = False,
        epochs: int = 400,
        num_query_points: int = 1,
        num_predict_samples: int = 100,
        noise_init: float = 1e-5
):

    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = noise_init

    svgp = build_vanilla_deep_gp(data, search_space, 1, num_inducing,
                                likelihood_variance=noise_variance, trainable_likelihood=learn_noise)

    upper_lengthscale_lim = 100*(search_space.upper - search_space.lower)*tf.sqrt(tf.cast(len(search_space.upper), dtype=tf.float64))
    lower_lengthscale_lim = upper_lengthscale_lim / 10000.
    svgp.f_layers[0].kernel.kernel = gpflow.kernels.Matern52(
        variance=svgp.f_layers[0].kernel.kernel.variance.numpy(),
        lengthscales=gpflow.Parameter(svgp.f_layers[0].kernel.kernel.lengthscales.numpy(),
                                      transform=tfp.bijectors.Sigmoid(low=lower_lengthscale_lim,
                                                                      high=upper_lengthscale_lim))
    )

    if fix_ips:
        set_trainable(svgp.f_layers[0].inducing_variable, False)

    acquisition_function = ParallelContinuousThompsonSampling()
    acquisition_rule = EfficientGlobalOptimization(
        acquisition_function,
        num_query_points=num_query_points,
        optimizer=split_acquisition_function_calls(
            generate_continuous_optimizer(
                1000 * tf.shape(search_space.lower)[-1],
                5 * tf.shape(search_space.lower)[-1],
                ),
            split_size=OPT_SPLIT_SIZE
        )
    )

    batch_size = 1000

    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            'loss', factor=0.5, patience=5, min_lr=1e-8, verbose=0
    )

    keras_optimizer = tf.optimizers.Adam(0.1, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": True,
        "callbacks": [scheduler]
    }
    optimizer = KerasOptimizer(keras_optimizer, fit_args)

    def predict_mean(query_points: TensorType, model: DeepGaussianProcess) -> TensorType:
        return model.predict(query_points)[0]

    return (
        DeepGaussianProcess(svgp, optimizer, continuous_optimisation=False),
        acquisition_rule,
        predict_mean
    )


def normalize(x, mean=None, std=None):
    if mean is None:
        mean = tf.math.reduce_mean(x, 0, True)
    if std is None:
        std = tf.math.sqrt(tf.math.reduce_variance(x, 0, True))
    return (x - mean) / std, mean, std
