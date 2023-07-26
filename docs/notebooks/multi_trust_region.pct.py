# %%
# from aim.ext.tensorboard_tracker import Run
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

import trieste
from trieste.acquisition import ParallelContinuousThompsonSampling
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.rule import MultiTrustRegionBox
from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_regret
from trieste.experimental.plotting.plotting import create_grid
from trieste.logging import pyplot
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import Hartmann6, ScaledBranin
from trieste.types import TensorType

# %%
np.random.seed(179)
tf.random.set_seed(179)

# CONFIG
tensorboard_dir_1 = f"./results/{datetime.now()}/tensorboard"

summary_writer = tf.summary.create_file_writer(tensorboard_dir_1)
trieste.logging.set_tensorboard_writer(summary_writer)

# %%
obj = ScaledBranin.objective
search_space = ScaledBranin.search_space


def obj_fun(
    x: TensorType,
) -> TensorType:  # contaminate observations with Gaussian noise
    return obj(x)  # + tf.random.normal([len(x), 1], 0, .1, tf.float64)


num_initial_data_points = 6
num_query_points = 3
num_steps = 10

# %%
initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.objectives.utils.mk_observer(obj_fun)
initial_data = observer(initial_query_points)


# gpflow_model = build_svgp(
#     initial_data, search_space, likelihood_variance=0.001, num_inducing_points=50
# )
#
# inducing_point_selector = ConditionalImprovementReduction()
#
# model = SparseVariational(
#     gpflow_model,
#     num_rff_features=1000,
#     inducing_point_selector=inducing_point_selector,
#     optimizer=BatchOptimizer(
#         tf.optimizers.Adam(0.05), max_iter=100, batch_size=50, compile=True
#     ),
# )
gpflow_model = build_gpr(
    initial_data,
    search_space,
    likelihood_variance=1e-4,
    trainable_likelihood=False,
)
model = GaussianProcessRegression(gpflow_model)

base_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
    builder=ParallelContinuousThompsonSampling(),
    num_query_points=num_query_points,
    optimizer=split_acquisition_function_calls(
        automatic_optimizer_selector, split_size=100_000
    ),
)

acq_rule = MultiTrustRegionBox(base_rule, number_of_tr=num_query_points)

ask_tell = AskTellOptimizer(
    search_space, initial_data, model, fit_model=True, acquisition_rule=acq_rule
)

# %%
color = cm.rainbow(np.linspace(0, 1, num_query_points))

Xplot, xx, yy = create_grid(
    mins=search_space.lower, maxs=search_space.upper, grid_density=90
)
ff = obj_fun(Xplot).numpy()

for step in range(num_steps):
    print(f"step number {step}")
    trieste.logging.set_step_number(step)

    new_points = ask_tell.ask()
    new_data = observer(new_points)
    # monitor models after each tell
    if summary_writer:
        models = ask_tell._models  # pylint: disable=protected-access
        trieste.logging.set_step_number(step)

        with summary_writer.as_default(step=step):
            for tag, model in models.items():
                with tf.name_scope(f"{tag}.model"):
                    model.log()

            fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(15, 5))
            fig.suptitle(f"step number {step}")
            ax[0, 0].scatter(
                ask_tell.dataset.query_points[:, 0].numpy(),
                ask_tell.dataset.query_points[:, 1].numpy(),
                color="blue",
            )
            ax[0, 0].scatter(
                new_points[:, 0].numpy(), new_points[:, 1].numpy(), color="red"
            )

            state = ask_tell.acquisition_state
            assert state is not None
            assert isinstance(state, MultiTrustRegionBox.State)

            xmin = {
                tag: state.acquisition_space.get_subspace(
                    tag
                ).get_local_min(  # type: ignore[attr-defined]
                    ask_tell.dataset
                )[
                    0
                ]
                for tag in state.acquisition_space.subspace_tags
            }
            i = 0

            ax[0, 1].contour(xx, yy, ff.reshape(*xx.shape), 80, alpha=0.5)

            for tag in state.acquisition_space.subspace_tags:
                ax[0, 1].scatter(
                    xmin[tag].numpy()[0],
                    xmin[tag].numpy()[1],
                    color=color[i],
                    marker="x",
                    alpha=0.5,
                )
                lb = state.acquisition_space.get_subspace(tag).lower
                ub = state.acquisition_space.get_subspace(tag).upper
                ax[0, 1].add_patch(
                    Rectangle(
                        (lb[0], lb[1]),
                        ub[0] - lb[0],
                        ub[1] - lb[1],
                        facecolor=color[i],
                        edgecolor=color[i],
                        alpha=0.3,
                    )
                )
                ax[0, 1].scatter(
                    new_points[i, 0].numpy(),
                    new_points[i, 1].numpy(),
                    color=color[i],
                    alpha=0.5,
                )
                ax[0, 1].scatter(
                    ask_tell.dataset.query_points[:, 0].numpy(),
                    ask_tell.dataset.query_points[:, 1].numpy(),
                    color="black",
                    alpha=0.2,
                )
                i = i + 1

            pyplot("Query points", fig)
            plt.close(fig)

    ask_tell.tell(new_data)

# %%
dataset = ask_tell.dataset

ground_truth_regret = obj(dataset.query_points) - Hartmann6.minimum
best_found_truth_idx = tf.squeeze(tf.argmin(ground_truth_regret, axis=0))

fig, ax = plt.subplots()
plot_regret(
    ground_truth_regret.numpy(), ax, num_init=10, idx_best=best_found_truth_idx
)

ax.set_yscale("log")
ax.set_ylabel("Regret")
ax.set_xlabel("# evaluations")

# %%
fig, ax = plt.subplots()
ax.scatter(dataset.query_points[:, 0], dataset.query_points[:, 1])
