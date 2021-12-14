# %% [markdown]
# # Bayesian active learning of failure or feasibility regions
#
# When designing a system it is important to identify design parameters that may affect the reliability of the system and cause failures, or lead to unsatisfactory performance. Consider designing a communication network that for some design parameters would lead to too long delays for users. A designer of the system would then decide what is the maximum acceptable delay and want to identify a *failure region* in the parameter space that would lead to longer delays., or conversely, a *feasible region* with safe performance.
#
# When evaluating the system is expensive (e.g. lengthy computer simulations), identification of the failure region needs to be performed with a limited number of evaluations. Traditional Monte Carlo based methods are not suitable here as they require too many evaluations. Bayesian active learning methods, however, are well suited for the task. Here we show how Trieste can be used to identify failure or feasible regions with the help of acquisition functions designed with this goal in mind.
#

# %%
# %matplotlib inline

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import numpy as np

np.random.seed(1793)
tf.random.set_seed(1793)


# %% [markdown]
# ## A toy problem
#
# Throughout the tutorial we will use the standard Branin function as a stand-in for an expensive-to-evaluate system. We create a failure region by thresholding the value at 80, space with value above 80 is considered a failure region. This region needs to be learned as efficiently as possible by the active learning algorithm.
#
# Note that if we are interested in a feasibility region instead, it is simply a complement of the failure region, space with the value below 80.
#
# We illustrate the thresholded Branin function below, you can note that above the threshold of 80 there are no more values observed.

# %%
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

search_space = BRANIN_SEARCH_SPACE

# # threshold is arbitrary, but has to be within the range of the function
# threshold = 80.0

# # define a modified branin function
# def thresholded_branin(x):
#     y = np.array(branin(x))
#     y[y > threshold] = np.nan
#     return tf.convert_to_tensor(y.reshape(-1, 1), x.dtype)


# # illustrate the thresholded branin function
# fig = plot_function_plotly(
#     thresholded_branin, search_space.lower, search_space.upper, grid_density=700
# )
# fig.update_layout(height=800, width=800)
# fig.show()


# %% [markdown]
# We start with a small initial dataset where our expensive-to-evaluate function is evaluated on points coming from a space-filling Halton sequence.

# %%
import trieste

# observer = trieste.objectives.utils.mk_observer(branin)

# num_initial_points = 6
# initial_query_points = search_space.sample_halton(num_initial_points)
# initial_data = observer(initial_query_points)





from trieste.objectives import (
    ACKLEY_5_MINIMIZER,
    ACKLEY_5_MINIMUM,
    ACKLEY_5_SEARCH_SPACE,
    BRANIN_MINIMIZERS,
    BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
    GRAMACY_LEE_MINIMIZER,
    GRAMACY_LEE_MINIMUM,
    GRAMACY_LEE_SEARCH_SPACE,
    HARTMANN_3_MINIMIZER,
    HARTMANN_3_MINIMUM,
    HARTMANN_3_SEARCH_SPACE,
    HARTMANN_6_MINIMIZER,
    HARTMANN_6_MINIMUM,
    HARTMANN_6_SEARCH_SPACE,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
    LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    MICHALEWICZ_5_MINIMIZER,
    MICHALEWICZ_5_MINIMUM,
    MICHALEWICZ_5_SEARCH_SPACE,
    MICHALEWICZ_10_MINIMIZER,
    MICHALEWICZ_10_MINIMUM,
    MICHALEWICZ_10_SEARCH_SPACE,
    ROSENBROCK_4_MINIMIZER,
    ROSENBROCK_4_MINIMUM,
    ROSENBROCK_4_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    SHEKEL_4_MINIMIZER,
    SHEKEL_4_MINIMUM,
    SHEKEL_4_SEARCH_SPACE,
    SIMPLE_QUADRATIC_MINIMIZER,
    SIMPLE_QUADRATIC_MINIMUM,
    TRID_10_MINIMIZER,
    TRID_10_MINIMUM,
    TRID_10_SEARCH_SPACE,
    ackley_5,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    michalewicz_2,
    michalewicz_5,
    michalewicz_10,
    rosenbrock_4,
    scaled_branin,
    shekel_4,
    simple_quadratic,
    trid_10,
)
from trieste.objectives.utils import mk_observer
import scipy


def get_volume(threshold, observer, search_space):
    estimate_query_points = search_space.sample(50000*search_space.dimension)
    estimate_data = observer(estimate_query_points)
    volume = tf.reduce_mean(tf.cast(estimate_data.observations > threshold, dtype=tf.float64))
    return volume.numpy()


def objective(threshold, target_volume, observer, search_space):
    estimate_query_points = search_space.sample(50000*search_space.dimension)
    estimate_data = observer(estimate_query_points)
    volume = tf.reduce_mean(tf.cast(estimate_data.observations > threshold, dtype=tf.float64))
    return (volume.numpy() - target_volume)**2


def find_threshold(target_volume, observer, search_space, minimum):

    estimate_query_points = search_space.sample_sobol(50000*search_space.dimension)
    estimate_data = observer(estimate_query_points)

    estimate_max = tf.reduce_max(estimate_data.observations)
    estimate_min = minimum[0]
    estimate_range = (estimate_min.numpy(), estimate_max.numpy())
    
    opt_results = scipy.optimize.brent(
        objective,
        args=(target_volume, observer, search_space),
        brack=estimate_range, 
        tol=1.48e-08, full_output=True, maxiter=500
    )
    print(opt_results)

    threshold = opt_results[0]

    volume = get_volume(threshold, observer, search_space)

    return threshold, volume



target_volume = 0.1

# find_threshold(
#     target_volume, mk_observer(branin), BRANIN_SEARCH_SPACE, BRANIN_MINIMUM
# )

# find_threshold(
#     target_volume, mk_observer(gramacy_lee), GRAMACY_LEE_SEARCH_SPACE, GRAMACY_LEE_MINIMUM
# )

# find_threshold(
#     target_volume, mk_observer(hartmann_3), HARTMANN_3_SEARCH_SPACE, HARTMANN_3_MINIMUM
# )

HARTMANN_6_THRESHOLD_10, _ = find_threshold(
    target_volume, mk_observer(hartmann_6), HARTMANN_6_SEARCH_SPACE, HARTMANN_6_MINIMUM
)

ACKLEY_5_THRESHOLD_10, _ = find_threshold(
    target_volume, mk_observer(ackley_5), ACKLEY_5_SEARCH_SPACE, ACKLEY_5_MINIMUM
)

# find_threshold(
#     target_volume, mk_observer(logarithmic_goldstein_price), LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE, LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM
# )

# find_threshold(
#     target_volume, mk_observer(michalewicz_2), MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM
# )


MICHALEWICZ_5_THRESHOLD_10, _ = find_threshold(
    target_volume, mk_observer(michalewicz_5), MICHALEWICZ_5_SEARCH_SPACE, MICHALEWICZ_5_MINIMUM
)

MICHALEWICZ_10_THRESHOLD_10, _ = find_threshold(
    target_volume, mk_observer(michalewicz_10), MICHALEWICZ_10_SEARCH_SPACE, MICHALEWICZ_10_MINIMUM
)

ROSENBROCK_4_THRESHOLD_10, _ = find_threshold(
    target_volume, mk_observer(rosenbrock_4), ROSENBROCK_4_SEARCH_SPACE, ROSENBROCK_4_MINIMUM
)

# SHEKEL_4_THRESHOLD_10, _ = find_threshold(
#     0.05, mk_observer(shekel_4), SHEKEL_4_SEARCH_SPACE, SHEKEL_4_MINIMUM
# )


threshold = HARTMANN_6_THRESHOLD_10
search_space = HARTMANN_6_SEARCH_SPACE
observer = mk_observer(hartmann_6)

num_initial_points = 10*search_space.dimension
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)



# %% [markdown]
# ## Probabilistic model of the objective function
#
# Just like in sequential optimization, we use a probabilistic model of the objective function. Acquisition functions will exploit the predictive posterior of the model to identify the failure region. We use a `GPR` model from the GPflow library to formulate a Gaussian process model, wrapping it in Trieste's `GaussianProcessRegression` model wrapper. As a good practice, we use priors for the kernel hyperparameters.

# %%
import gpflow
from trieste.models.gpflow.models import GaussianProcessRegression
import tensorflow_probability as tfp


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    dims = data.query_points.shape[-1]
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2]*dims)
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(variance), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)


# %% [markdown]
# ## Active learning with Expected feasibility acquisition function
#
# The problem of identifying a failure or feasibility region of a (expensive-to-evaluate) function $f$ can be formalized as estimating the excursion set, $\Gamma^* = \{ x \in X: f(x) \ge T\}$, or estimating the contour line, $C^* = \{ x \in X: f(x) = T\}$, for some threshold $T$ (see <cite data-cite="bect2012sequential"/> for more details).
#
# It turns out that Gaussian processes can be used as classifiers for identifying where excursion probability is larger than 1/2 and this idea is used to build many sequential sampling strategies. Here we introduce Expected feasibility acquisition function that implements two related sampling strategies called *bichon* criterion (<cite data-cite="bichon2008efficient"/>) and *ranjan* criterion (<cite data-cite="ranjan2008sequential"/>). <cite data-cite="bect2012sequential"/> provides a common expression for these two criteria: $$\mathbb{E}[\max(0, (\alpha s(x))^\delta - |T - m(x)|^\delta)]$$
#
# Here $m(x)$ and $s(x)$ are the mean and standard deviation of the predictive posterior of the Gaussian process model. Bichon criterion is obtained when $\delta = 1$ while ranjan criterion is obtained when $\delta = 2$. $\alpha>0$ is another parameter that acts as a percentage of standard deviation of the posterior around the current boundary estimate where we want to sample. The goal is to sample a point with a mean close to the threshold $T$ and a high variance, so that the positive difference in the equation above is as large as possible.


# %% [markdown]
# We now illustrate `ExpectedFeasibility` acquisition function using the Bichon criterion. Performance for the Ranjan criterion is typically very similar. `ExpectedFeasibility` takes threshold as an input and has two parameters, `alpha` and `delta` following the description above.
#
# Note that even though we use below `ExpectedFeasibility` with `EfficientGlobalOptimization`  `BayesianOptimizer` routine, we are actually performing active learning. The only relevant difference between the two is the nature of the acquisition function - optimization ones are designed with the goal of finding the optimum of a function, while active learning ones are designed to learn the function (or some aspect of it, like here).

# %%
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedFeasibility

# Bichon criterion
delta = 1

# # set up the acquisition rule and initialize the Bayesian optimizer
# acq = ExpectedFeasibility(threshold, delta=delta)
# rule = EfficientGlobalOptimization(builder=acq)  # type: ignore
# bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

# num_steps = 20
# result = bo.optimize(num_steps, initial_data, model, rule)




# num_search_space_samples = 10000
# num_query_points = 10*search_space.dimension

# acq_rule = trieste.acquisition.rule.DiscreteThompsonSampling(
#     num_search_space_samples=num_search_space_samples,
#     num_query_points=num_query_points,
#     thompson_sampler = trieste.acquisition.sampler.ExactThompsonSampler(
#         sample_min_value=False, threshold=threshold)
# )

# bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

# num_steps = 20
# result = bo.optimize(
#     num_steps, initial_data, model, acq_rule, track_state=True
# )



num_search_space_samples = 10000
num_query_points = 10*search_space.dimension

acq_rule = trieste.acquisition.rule.DiscreteThompsonSampling(
    num_search_space_samples=num_search_space_samples,
    num_query_points=num_query_points,
    thompson_sampler = trieste.acquisition.sampler.ExpectedFeasibilitySampler(
        threshold=threshold, alpha=1, delta=1)
)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 20
result = bo.optimize(
    num_steps, initial_data, model, acq_rule, track_state=True
)






from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import BRANIN_SEARCH_SPACE, branin, scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.observer import Observer
from trieste.space import Box, SearchSpace
from trieste.types import TensorType


def _excursion_probability(
    x: TensorType, model: TrainableProbabilisticModel, threshold: float
) -> tfp.distributions.Distribution:
    mean, variance = model.model.predict_f(x)  # type: ignore
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)


def _get_excursion_accuracy(
    x: TensorType, model: TrainableProbabilisticModel, threshold: float
) -> float:
    prob = _excursion_probability(x, model, threshold)
    accuracy = tf.reduce_sum(prob * (1 - prob))

    return accuracy


def _get_feasible_set_test_data(
    search_space: Box,
    observer: Observer,
    n_global: int,
    n_boundary: int,
    threshold: float,
    range_pct: float = 0.01,
):

    boundary_done = False
    global_done = False
    boundary_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))
    global_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))

    while not boundary_done and not global_done:
        test_query_points = search_space.sample(100000)
        test_data = observer(test_query_points)
        threshold_deviation = range_pct * (
            tf.reduce_max(test_data.observations)  # type: ignore
            - tf.reduce_min(test_data.observations)  # type: ignore
        )

        mask = tf.reduce_all(
            tf.concat(
                [
                    test_data.observations > threshold - threshold_deviation,  # type: ignore
                    test_data.observations < threshold + threshold_deviation,  # type: ignore
                ],
                axis=1,
            ),
            axis=1,
        )
        boundary_points = tf.concat(
            [boundary_points, tf.boolean_mask(test_query_points, mask)], axis=0
        )
        global_points = tf.concat(
            [global_points, tf.boolean_mask(test_query_points, tf.logical_not(mask))], axis=0
        )

        if boundary_points.shape[0] > n_boundary:
            boundary_done = True
        if global_points.shape[0] > n_global:
            global_done = True

    return (
        global_points[
            :n_global,
        ],
        boundary_points[
            :n_boundary,
        ],
    )


# we set a performance criterion at 0.001 probability of required precision per point
# for global points and 0.01 close to the boundary
n_global = 10000 * search_space.dimension
n_boundary = 2000 * search_space.dimension
global_test, boundary_test = _get_feasible_set_test_data(
    search_space, observer, n_global, n_boundary, threshold, range_pct=0.03
)

global_criterion = 0.001 * (1 - 0.001) * tf.cast(n_global, tf.float64)
boundary_criterion = 0.01 * (1 - 0.01) * tf.cast(n_boundary, tf.float64)


# we expect a model with initial data to fail the criteria
initial_model = build_model(initial_data)
initial_model.optimize(initial_data)
initial_accuracy_global = _get_excursion_accuracy(global_test, initial_model, threshold)
initial_accuracy_boundary = _get_excursion_accuracy(boundary_test, initial_model, threshold)
print((global_criterion.numpy(), boundary_criterion.numpy()))
print((initial_accuracy_global.numpy(), initial_accuracy_boundary.numpy()))


# after active learning the model should be much more accurate
final_model = result.try_get_final_model()
final_accuracy_global = _get_excursion_accuracy(global_test, final_model, threshold)
final_accuracy_boundary = _get_excursion_accuracy(boundary_test, final_model, threshold)

print((global_criterion.numpy(), boundary_criterion.numpy()))
print((final_accuracy_global.numpy(), final_accuracy_boundary.numpy()))



accuracy_global = []
accuracy_boundary = []
for step in range(len(result.history)):
    print(step)
    inter_model = result.history[step].models['OBJECTIVE']
    accuracy_global.append(_get_excursion_accuracy(global_test, inter_model, threshold).numpy())
    accuracy_boundary.append(_get_excursion_accuracy(boundary_test, inter_model, threshold).numpy())
final_model = result.try_get_final_model()
accuracy_global.append(_get_excursion_accuracy(global_test, final_model, threshold).numpy())
accuracy_boundary.append(_get_excursion_accuracy(boundary_test, final_model, threshold).numpy())


# accuracy_global = [10507.551034522772,
#  2052.8754196255445,
#  1240.2070146687079,
#  913.5873631321928,
#  535.6929315731616,
#  347.5203001353799,
#  250.08394186513266,
#  169.95180541239657,
#  128.26677975625674,
#  101.39667495590126,
#  75.99732494224494,
#  48.50559445358459,
#  39.810157192123505,
#  29.143975116294214,
#  24.13614190988836,
#  22.22347561131885,
#  18.66782684497504,
#  16.44038038861271,
#  15.371334816611629,
#  14.07201219773767,
#  11.551616435948397]

# accuracy_boundary = [2943.9005571522866,
#  2390.622895196025,
#  2204.8608959164503,
#  2111.655806702366,
#  1968.6308032954155,
#  1842.7822534912661,
#  1779.6961127799273,
#  1696.3737275370172,
#  1621.6972530307887,
#  1574.6310867145653,
#  1520.8448991649404,
#  1471.8827226671451,
#  1419.3650526411516,
#  1367.5361768030666,
#  1338.082243169053,
#  1300.9694607880642,
#  1263.8875399166805,
#  1239.984372863929,
#  1206.0357007794714,
#  1185.8794191762122,
#  1160.6155325134373]


import matplotlib.pyplot as plt

fig, axis = plt.subplots(
    1, 2, squeeze=False, sharex=True,
)
axis[0, 0].plot(range(0,num_steps+1), accuracy_global)
axis[0, 0].set_title("Global accuracy")
plt.xticks(range(0,num_steps+1,2))

axis[0, 1].plot(range(0,num_steps+1), accuracy_boundary)
axis[0, 1].set_title("Boundary accuracy")

plt.show()










# %% [markdown]
# Let's illustrate the results.
#
# To identify the failure or feasibility region we compute the excursion probability using our Gaussian process model: $$P\left(\frac{m(x) - T}{s(x)}\right)$$ where $m(x)$ and $s(x)$ are the mean and standard deviation of the predictive posterior of the model given the data.
#
# We plot a two-dimensional contour map of our thresholded Branin function as a reference, excursion probability map using the model fitted to the initial data alone, and updated excursion probability map after all the active learning steps.
#
# We first define helper functions for computing excursion probabilities and plotting, and then plot the thresholded Branin function as a reference. White area represents the failure region.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm

from trieste.acquisition import AcquisitionFunction
from trieste.types import TensorType
from trieste.utils import to_numpy
from trieste.acquisition.multi_objective.dominance import non_dominated

from util.plotting import create_grid, plot_surface, format_point_markers


def plot_bo_points(
    pts,
    ax,
    num_init=None,
    idx_best=None,
    mask_fail=None,
    obs_values=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_best="tab:purple",
):
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to BO factors.
    :param pts: [N, 2] x inputs
    :param ax: a plt axes object
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param obs_values: optional [N] outputs (for 3d plots)
    """

    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_best, mask_fail, m_init, m_add, c_pass, c_fail, c_best
    )

    if obs_values is None:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], facecolors='none', edgecolors=col_pts[i], marker=mark_pts[i])
    else:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], obs_values[i], facecolors='none', edgecolors=col_pts[i], marker=mark_pts[i])


def excursion_probability(x, model, threshold=80):
    mean, variance = model.model.predict_f(x)
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    threshold = tf.cast(threshold, x.dtype)

    if tf.size(threshold) == 1:
        t = (mean - threshold) / tf.sqrt(variance)
        return normal.cdf(t)
    else:
        t0 = (mean - threshold[0]) / tf.sqrt(variance)
        t1 = (mean - threshold[1]) / tf.sqrt(variance)
        return normal.cdf(t1) - normal.cdf(t0)


def plot_dims(
    dims,
    search_space,
    model,
    threshold,
    title,
    query_points,
    num_initial_points,
    figsize,
    fill=True,
    alpha=1.0
):

    def thresholded_observer(x):
        y = to_numpy(observer(points).observations)
        y[y > threshold] = np.nan
        return y.reshape(-1, 1)

    def objective_function(x):
        return excursion_probability(x, model, threshold)

    mins = to_numpy(search_space.lower)
    maxs = to_numpy(search_space.upper)
    test_query_points = search_space.sample(100**2)

    xspaced = np.linspace(mins[0], maxs[0], 100)
    yspaced = np.linspace(mins[1], maxs[1], 100)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T

    points = test_query_points.numpy()
    points[:,dims[0]] = Xplot[:,0]
    points[:,dims[1]] = Xplot[:,1]

    value = to_numpy(objective_function(points))
    value = value.reshape(-1, 1)

    obj_value = thresholded_observer(points)

    xlabel="$X_" + str(dims[0]) + "$"
    ylabel="$X_" + str(dims[1]) + "$" 

    fig, ax = plt.subplots(
        1, 2, squeeze=False, sharex="all", sharey="all", figsize=figsize
    )

    # objective function
    plt_obj = plot_surface(xx, yy, obj_value, ax[0, 0], contour=True, alpha=alpha, fill=fill)
    ax[0, 0].set_title(title[0])
    fig.colorbar(plt_obj, ax=ax[0, 0])
    ax[0, 0].set_xlabel(xlabel)
    ax[0, 0].set_ylabel(ylabel)
    ax[0, 0].set_xlim(mins[dims[0]], maxs[dims[0]])
    ax[0, 0].set_ylim(mins[dims[1]], maxs[dims[1]])

    # exc probability based on the model
    plt_obj = plot_surface(xx, yy, value, ax[0, 1], contour=True, alpha=alpha, fill=fill)
    ax[0, 1].set_title(title[1])
    fig.colorbar(plt_obj, ax=ax[0, 1])
    ax[0, 1].set_xlabel(xlabel)
    ax[0, 1].set_ylabel(ylabel)
    ax[0, 1].set_xlim(mins[dims[0]], maxs[dims[0]])
    ax[0, 1].set_ylim(mins[dims[1]], maxs[dims[1]])
    plot_bo_points(query_points, ax[0, 1], num_initial_points, c_pass="tab:red")

    return fig, ax


dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

for i in range(0,6-1):
    for j in range(i+1,6):
        plot_dims(
            (i,j),
            search_space,
            final_model,
            threshold,
            ["Thresholded Hartmann 6d (10% volume)", "Excursion probability"],
            query_points,
            num_initial_points,
            (15,8)
        )
        # plt.show()
        plt.savefig('hartmann6_10_' + str(i) + '_' + str(j) +'.png', dpi=800)













