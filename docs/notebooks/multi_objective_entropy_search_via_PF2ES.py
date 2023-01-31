# %% [markdown]
# # Multi-Objective Optimization (MOO) using \{PF\}$^2$ES

# %% [markdown]
# In this tutorial, we demonstrate how to utilize our proposed new information-theoretic acquisition function: \{PF\}$^2$ES (https://arxiv.org/abs/2204.05411) for multi-objective optimization (MOO).
#
# \{PF\}$^2$ES is suitable for:
# - Observation Noise Free MOO problem by sequential sampling / parallel sampling
# - Observation Noise Free C(onstraint)MOO problem by sequential sampling / parallel sampling
#
# Note we follow the convention of trieste by performing **minimization** below. 
#

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# For a fully demonstration of \{PF\}$^2$ES's utility, besides using the ` BayesianOptimizer` as a defualt choice, we recommend using an `Ask-Tell` interface, since it provides the functionality of visualizing the intermediate Pareto frontiers samples generated acquisition function, which is helpful for us to understand the uncertainty of the Pareto frontier. We provide a modified `AskTellOptimizer` below to return the intermediate data that we would like to inspect.

# %%
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.types import TensorType
from typing import Tuple

class AskTellOptimizer_with_PF_inference(AskTellOptimizer):
    """
    Here, we construct a modified AskTell Interface so that  in each (Batch) iteration, we are able to see
    how the current GP inferred Pareto Frontier looks like, this can give us a hint that how the uncertainty interms
    of Pareto frontier is!
    """

    def ask(self) -> Tuple[TensorType, TensorType]:
        """Suggests a point (or points in batch mode) to observe by optimizing the acquisition
        function. If the acquisition is stateful, its state is saved.

        :return: A :class:`TensorType` instance representing suggested point(s).
        """
        # This trick deserves a comment to explain what's going on
        # acquisition_rule.acquire can return different things:
        # - when acquisition has no state attached, it returns just points
        # - when acquisition has state, it returns a Callable
        #   which, when called, returns state and points
        # so code below is needed to cater for both cases

        with Timer() as query_point_generation_timer:
            points_or_stateful = self._acquisition_rule.acquire(
                self._search_space, self._models, datasets=self._datasets
            )

        if callable(points_or_stateful):
            self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
        else:
            query_points = points_or_stateful

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                if tf.rank(query_points) == 2:
                    for i in tf.range(tf.shape(query_points)[1]):
                        if len(query_points) == 1:
                            logging.scalar(f"query_points/[{i}]", float(query_points[0, i]))
                        else:
                            logging.histogram(f"query_points/[{i}]", query_points[:, i])
                    logging.histogram(
                        "query_points/euclidean_distances", lambda: pdist(query_points)
                    )
                logging.scalar(
                    "wallclock/query_point_generation",
                    query_point_generation_timer.time,
                )

        return query_points, ask_tell._acquisition_rule._builder._pf_samples


# %% [markdown]
# ## Demonstration on MOO problem

# %% [markdown]
# ### Problem Definition

# %% [markdown]
# We consider the VLMOP2 function --- a synthetic benchmark problem with two objectives. We start by defining the problem parameters.

# %%
import timeit

import gpflow
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist

from trieste import logging
from trieste.acquisition.function.multi_objective import PF2ES
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.experimental.plotting import (
    plot_bo_points,
    plot_function_2d,
    plot_mobo_points_in_obj_space,
)
from trieste.models import TrainableHasTrajectoryAndPredictJointReparamModelStack
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.observer import OBJECTIVE
from trieste.utils import Timer

# %%
import trieste
from trieste.space import Box, SearchSpace
from trieste.objectives.multi_objectives import VLMOP2

vlmop2 = VLMOP2(input_dim=2).objective
observer = trieste.objectives.utils.mk_observer(vlmop2)

# %%
mins = [-2, -2]
maxs = [2, 2]
vlmop2_search_space = Box(mins, maxs)
vlmop2_num_objective = 2

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 5  # We use 2d+1 as initial doe number
initial_query_points = vlmop2_search_space.sample(num_initial_points)
vlmop2_initial_data = observer(initial_query_points)

# %% [markdown]
# ... and visualise the data across the design space: each figure contains the contour lines of each objective function.

# %%
_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    title=["Obj 1", "Obj 2"],
    figsize=(12, 6),
    colorbar=True,
    xlabel="$X_1$",
    ylabel="$X_2$",
)
plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()


# %%
def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(
            single_obj_data, search_space, likelihood_variance=1e-7, trainable_likelihood=False
        )
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)


# %%
vlmop2_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    vlmop2_initial_data, vlmop2_num_objective, vlmop2_search_space
)
ask_tell = AskTellOptimizer_with_PF_inference(
    vlmop2_search_space,
    vlmop2_initial_data,
    vlmop2_model,
    acquisition_rule=EfficientGlobalOptimization(PF2ES(vlmop2_search_space)),
)

# %% [markdown]
# ### Sequential MOO by \{PF\}$^2$ES

# %% [markdown]
# We now conduct multi-objective optimization on VLMOP2 based on our sequential \{PF\}$^2$ES, the whole below process may takes around 6-10 minutes depending on the setting of `n_steps`.
#
# We note since we demonstrate the plot investigation of intermediate results within each BO iter here, this consumes larger time than just performing BO for \{PF\}$^2$ES, which's acquiring time has been printed below and only takes around 4 mins.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 20

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_mean, pred_var = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im1 = axs[0].scatter(
        *tf.split(pred_mean, 2, axis=-1), label="Predicted BO Sample Data", color="orange"
    )
    ellipse = Ellipse(
        (pred_mean[0, 0], pred_mean[0, 1]),
        2 * tf.sqrt(pred_var[0, 0]),
        2 * tf.sqrt(pred_var[0, 1]),
        angle=0,
        alpha=0.2,
        edgecolor="k",
    )
    im2 = axs[0].add_artist(ellipse)

    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO samples 
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        vlmop2_model.get_internal_data().observations, ax4plot=axs[1], return_path_collections=True
    )

    ims.append(_ims + [im1, im2, im3] + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
data_query_points = ask_tell.datasets[OBJECTIVE].query_points
data_observations = ask_tell.datasets[OBJECTIVE].observations

_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    figsize=(12, 6),
    title=["Obj 1", "Obj 2"],
    xlabel="$X_1$",
    ylabel="$X_2$",
    colorbar=True,
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
from trieste.acquisition.multi_objective.utils import inference_pareto_fronts_from_gp_mean

rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    vlmop2_model, vlmop2_search_space, popsize=50
)

real_pf = vlmop2(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
plot_mobo_points_in_obj_space(real_pf, ax4plot=ax)
ax.scatter(
    *tf.split(VLMOP2(input_dim=2).gen_pareto_optimal_points(50), 2, -1), label="reference Pareto Frontier", s=5
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()

# %% [markdown]
# ### Batched MOO by q- \{PF\}$^2$ES

# %% [markdown]
# We now conduct show that \{PF\}$^2$ES can also conduct batch multi-objective optimization (we referred to as q-\{PF\}$^2$ES), given parallel computation resources, this allow taking smaller BO iterations while achiving similar performance result on optimal Pareto frontier recommendation.
#
# We note the only change to enable q- \{PF\}$^2$ES is by specifying `parallel_sampling=True` to let the acquisition function know its performed in a batch setting, we can also specify the Monte Carlo sample size to 128 by using `batch_mc_sample_size=128` (here for speed we only use 64), eventually, we set `num_query_points=2`as we would like to sample 2 points in a batch.

# %%
vlmop2_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    vlmop2_initial_data, vlmop2_num_objective, vlmop2_search_space
)
ask_tell = AskTellOptimizer_with_PF_inference(
    vlmop2_search_space,
    vlmop2_initial_data,
    vlmop2_model,
    acquisition_rule=EfficientGlobalOptimization(
        PF2ES(vlmop2_search_space, parallel_sampling=True, batch_mc_sample_size=64),
        num_query_points=2,
    ),
)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 10

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")
axs[0].legend()
axs[1].legend()
axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_means, pred_vars = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer(new_point)
    ask_tell.tell(new_data)
    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im12s = []
    for pred_mean, pred_var in zip(pred_means, pred_vars):
        # print(pred_mean)
        im12s.append(
            axs[0].scatter(
                *tf.split(pred_mean, 2, axis=-1), color="r", label="Predicted BO Sample Data"
            )
        )
        ellipse = Ellipse(
            (pred_mean[0], pred_mean[1]),
            2 * tf.sqrt(pred_var[0]),
            2 * tf.sqrt(pred_var[1]),
            angle=0,
            alpha=0.2,
            edgecolor="k",
        )
        im12s.append(axs[0].add_artist(ellipse))
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    im12s.append(im3)
    # plot actual BO samples 
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        vlmop2_model.get_internal_data().observations, ax4plot=axs[1], return_path_collections=True
    )

    ims.append(_ims + im12s + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
data_query_points = ask_tell.datasets[OBJECTIVE].query_points
data_observations = ask_tell.datasets[OBJECTIVE].observations
plt.figure()
_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    figsize=(12, 6),
    title=["Obj 1", "Obj 2"],
    xlabel="$X_1$",
    ylabel="$X_2$",
    colorbar=True,
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
from trieste.acquisition.multi_objective.utils import inference_pareto_fronts_from_gp_mean

rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    vlmop2_model, vlmop2_search_space, popsize=50
)

real_pf = vlmop2(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
plot_mobo_points_in_obj_space(real_pf, ax4plot=ax)
ax.scatter(
    *tf.split(VLMOP2(input_dim=2).gen_pareto_optimal_points(50), 2, -1), label="reference Pareto Frontier", s=5
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()

# %% [markdown]
# -----------------

# %% [markdown]
# ## Demonstration on  CMOO problem

# %% [markdown]
# ### Problem Definition

# %% [markdown]
# We now demonstrate that \{PF\}$^2$ES and its parallel version are also able to perform Constraint MOO (CMOO) problem.

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
CONSTRAINT = "CONSTRAINT"

# %%
num_initial_points = 5
cvlmop2_num_objective = 2
cvlmop2_num_constraints = 1


class Sim:
    threshold = 0.0

    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return -(
            z[:, None] - 0.75
        )  # take the inverse since we assume constraint < 0  means infeasible


OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"


def observer_cst(query_points):
    return {
        OBJECTIVE: Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: Dataset(query_points, Sim.constraint(query_points)),
    }


cvlmop2_initial_query_points = vlmop2_search_space.sample(num_initial_points)
cvlmop2_initial_data_with_cst = observer_cst(cvlmop2_initial_query_points)
cvlmop2_search_space = vlmop2_search_space

# %%
from trieste.experimental.plotting import plot_2obj_cst_query_points

plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    cvlmop2_initial_data_with_cst[OBJECTIVE].astuple(),
    cvlmop2_initial_data_with_cst[CONSTRAINT].astuple(),
    larger_feasible=True,  # our formulation assume >=0  is feasible
)
plt.show()

mask_fail = cvlmop2_initial_data_with_cst[CONSTRAINT].observations.numpy() < 0
plot_mobo_points_in_obj_space(
    cvlmop2_initial_data_with_cst[OBJECTIVE].observations, mask_fail=mask_fail[:, 0]
)
plt.show()

# %% [markdown]
# ### Sequential MOO by \{PF\}$^2$ES

# %%
import copy  # for not changing cvlmop2_initial_data_with_cst, which is reused in Batch CMOO

cvlmop2_model = {
    OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[OBJECTIVE]),
        num_output=cvlmop2_num_objective,
        search_space=cvlmop2_search_space,
    ),
    CONSTRAINT: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[CONSTRAINT]),
        num_output=cvlmop2_num_constraints,
        search_space=cvlmop2_search_space,
    ),
}

ask_tell = AskTellOptimizer_with_PF_inference(
    cvlmop2_search_space,
    copy.deepcopy(cvlmop2_initial_data_with_cst),
    cvlmop2_model,
    acquisition_rule=EfficientGlobalOptimization(
        PF2ES(cvlmop2_search_space, constraint_tag=CONSTRAINT)
    ),
)

# %% [markdown]
# We now conduct CMOO on constraint-VLMOP2 based on our sequential \{PF\}$^2$ES, the whole below process may takes around 6-10 minutes depending on the setting of `n_steps`. Note the experiments conducted below may take 20 minutes

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 30

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_mean, pred_var = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer_cst(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im1 = axs[0].scatter(
        *tf.split(pred_mean, 2, axis=-1), label="Predicted BO Sample Data", color="orange"
    )
    ellipse = Ellipse(
        (pred_mean[0, 0], pred_mean[0, 1]),
        2 * tf.sqrt(pred_var[0, 0]),
        2 * tf.sqrt(pred_var[0, 1]),
        angle=0,
        alpha=0.2,
        edgecolor="k",
    )
    im2 = axs[0].add_artist(ellipse)
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO
    mask_fail = ask_tell._models[CONSTRAINT].get_internal_data().observations.numpy() < 0
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        cvlmop2_model[OBJECTIVE].get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
        mask_fail=mask_fail[:, 0],
    )

    ims.append(_ims + [im1, im2, im3] + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
objective_dataset = ask_tell.datasets[OBJECTIVE]
constraint_dataset = ask_tell.datasets[CONSTRAINT]

plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    objective_dataset.astuple(),
    constraint_dataset.astuple(),
    larger_feasible=True,  # our formulation assume >=0  is feasible
)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
from trieste.acquisition.multi_objective.utils import (
    inference_pareto_fronts_from_gp_mean,
    moo_nsga2_pymoo,
)

# Out-of-sample recommendation
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    cvlmop2_model[OBJECTIVE],
    cvlmop2_search_space,
    popsize=50,
    cons_models=cvlmop2_model[CONSTRAINT],
    min_feasibility_probability=0.95,
    constraint_enforce_percentage=5e-3,
)


# we generate the reference feasible Pareto frontier based on the real problem
reference_pf = moo_nsga2_pymoo(
    Sim().objective,
    input_dim=2,
    obj_num=2,
    bounds=tf.convert_to_tensor(VLMOP2(input_dim=2).bounds),
    popsize=50,
    cons=Sim().constraint,
    cons_num=1,
)

# %%
rec_actual_obs_datasets = observer_cst(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
mask_fail = rec_actual_obs_datasets[CONSTRAINT].observations.numpy() < 0
plot_mobo_points_in_obj_space(
    rec_actual_obs_datasets[OBJECTIVE].observations, mask_fail=mask_fail[:, 0], ax4plot=ax
)
ax.scatter(*tf.split(reference_pf.fronts, 2, -1), label="reference Pareto Frontier", s=5)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()


# %% [markdown]
# ### Batch CMOO by q- \{PF\}$^2$ES

# %% [markdown]
# Eventually, we use q-\{PF\}$^2$ES) for CMOO. Note the whole experiments may take around 20 min.

# %%
import copy

cvlmop2_model = {
    OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[OBJECTIVE]),
        num_output=cvlmop2_num_objective,
        search_space=cvlmop2_search_space,
    ),
    CONSTRAINT: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[CONSTRAINT]),
        num_output=cvlmop2_num_constraints,
        search_space=cvlmop2_search_space,
    ),
}

ask_tell = AskTellOptimizer_with_PF_inference(
    cvlmop2_search_space,
    copy.deepcopy(cvlmop2_initial_data_with_cst),
    cvlmop2_model,
    acquisition_rule=EfficientGlobalOptimization(
        PF2ES(
            cvlmop2_search_space,
            constraint_tag=CONSTRAINT,
            parallel_sampling=True,
            batch_mc_sample_size=64,
        ),
        num_query_points=2,
    ),
)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 15

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_means, pred_vars = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer_cst(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im12s = []
    for pred_mean, pred_var in zip(pred_means, pred_vars):
        # print(pred_mean)
        im12s.append(
            axs[0].scatter(
                *tf.split(pred_mean, 2, axis=-1), color="r", label="Predicted BO Sample Data"
            )
        )
        ellipse = Ellipse(
            (pred_mean[0], pred_mean[1]),
            2 * tf.sqrt(pred_var[0]),
            2 * tf.sqrt(pred_var[1]),
            angle=0,
            alpha=0.2,
            edgecolor="k",
        )
        im12s.append(axs[0].add_artist(ellipse))
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO
    mask_fail = ask_tell._models[CONSTRAINT].get_internal_data().observations.numpy() < 0
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        cvlmop2_model[OBJECTIVE].get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
        mask_fail=mask_fail[:, 0],
    )

    ims.append(_ims + im12s + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
objective_dataset = ask_tell.datasets[OBJECTIVE]
constraint_dataset = ask_tell.datasets[CONSTRAINT]
data_query_points = objective_dataset.query_points
data_observations = objective_dataset.observations

plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    objective_dataset.astuple(),
    constraint_dataset.astuple(),
    larger_feasible=True,  # our formulation assume >=0  is feasible
)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
from trieste.acquisition.multi_objective.utils import (
    inference_pareto_fronts_from_gp_mean,
    moo_nsga2_pymoo,
)

# Out-of-sample recommendation
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    cvlmop2_model[OBJECTIVE],
    cvlmop2_search_space,
    popsize=50,
    cons_models=cvlmop2_model[CONSTRAINT],
    min_feasibility_probability=0.95,
    constraint_enforce_percentage=5e-3,
)


# we generate the reference feasible Pareto frontier based on the real problem
reference_pf = moo_nsga2_pymoo(
    Sim().objective,
    input_dim=2,
    obj_num=2,
    bounds=tf.convert_to_tensor(VLMOP2(input_dim=2).bounds),
    popsize=50,
    cons=Sim().constraint,
    cons_num=1,
)

# %%
rec_actual_obs_datasets = observer_cst(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
mask_fail = rec_actual_obs_datasets[CONSTRAINT].observations.numpy() < 0
plot_mobo_points_in_obj_space(
    rec_actual_obs_datasets[OBJECTIVE].observations, mask_fail=mask_fail[:, 0], ax4plot=ax
)
ax.scatter(*tf.split(reference_pf.fronts, 2, -1), label="reference Pareto Frontier", s=5)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()
