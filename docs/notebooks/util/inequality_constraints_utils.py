# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import numpy as np

from .plotting import create_grid


def plot_objective_and_constraints(search_space, simulation):
    objective_fn = simulation.objective
    constraint_fn = simulation.constraint
    lower_bound = search_space.lower
    upper_bound = search_space.upper

    grid, xx, yy = create_grid(lower_bound, upper_bound, grid_density=80)
    objective = objective_fn(grid).numpy()
    constraint = constraint_fn(grid).numpy()
    fig, (axes1, axes2) = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(6, 6))

    levels = 30

    axes1[0].contourf(xx, yy, objective.reshape(*xx.shape), levels, alpha=0.9)
    axes1[1].contourf(xx, yy, constraint.reshape(*xx.shape), levels, alpha=0.9)
    axes1[0].set_title("Objective")
    axes1[1].set_title("Constraint")

    mask_ids = np.argwhere(constraint > simulation.threshold)
    mask = np.zeros_like(objective, dtype=bool)
    mask[mask_ids] = True
    objective_masked = np.ma.array(objective, mask=mask)
    constraint_masked = np.ma.array(constraint, mask=mask)

    axes2[0].contourf(xx, yy, objective_masked.reshape(*xx.shape), levels, alpha=0.9)
    axes2[1].contourf(xx, yy, constraint_masked.reshape(*xx.shape), levels, alpha=0.9)
    axes2[0].set_title("Constrained objective")
    axes2[1].set_title("Constraint mask")

    for ax in np.ravel([axes1, axes2]):
        ax.set_xlim(lower_bound[0], upper_bound[0])
        ax.set_ylim(lower_bound[1], upper_bound[1])

    return fig


def plot_init_query_points(
    search_space, simulation, objective_data, constraint_data, new_constraint_data=None
):
    objective_fn = simulation.objective
    constraint_fn = simulation.constraint
    lower_bound = search_space.lower
    upper_bound = search_space.upper
    levels = 30
    psize = 15
    cw, cb, co = "white", "tab:blue", "tab:orange"

    grid, xx, yy = create_grid(lower_bound, upper_bound, grid_density=80)
    objective = objective_fn(grid).numpy()
    constraint = constraint_fn(grid).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    constraint_points = constraint_data[-1].numpy()
    x = objective_data[0].numpy()

    mask_ids = constraint[:, 0] > simulation.threshold
    mask = np.zeros_like(objective, dtype=bool)
    mask[mask_ids, :] = True
    objective_masked = np.ma.array(objective, mask=mask)

    def in_out_points(x, constraint_points):
        ids_in = constraint_points[:, 0] <= simulation.threshold
        ids_out = constraint_points[:, 0] > simulation.threshold
        return x.T[..., ids_in], x.T[..., ids_out]

    (x_in, y_in), (x_out, y_out) = in_out_points(x, constraint_points)

    ax.contourf(xx, yy, objective_masked.reshape(*xx.shape), levels, alpha=0.9)
    ax.scatter(x_in, y_in, s=psize, c=cb, edgecolors=cw, marker="o")
    ax.scatter(x_out, y_out, s=psize, c=cw, edgecolors=cb, marker="o")

    if new_constraint_data is not None:
        x_new, constraint_points_new = new_constraint_data
        (x_in_new, y_in_new), (x_out_new, y_out_new) = in_out_points(
            x_new.numpy(), constraint_points_new.numpy()
        )
        ax.scatter(x_in_new, y_in_new, s=psize, c=co, edgecolors=cw, marker="o")
        ax.scatter(x_out_new, y_out_new, s=psize, c=cw, edgecolors=co, marker="o")

    ax.set_title("Constrained objective")
    ax.set_xlim(lower_bound[0], upper_bound[0])
    ax.set_ylim(lower_bound[1], upper_bound[1])

    return fig


def plot_2obj_cst_query_points(search_space, simulation, objective_data, constraint_data):
    class Sim1(simulation):
        @staticmethod
        def objective(input_data):
            return simulation.objective(input_data)[:, 0:1]

    class Sim2(simulation):
        @staticmethod
        def objective(input_data):
            return simulation.objective(input_data)[:, 1:2]

    for sim in [Sim1, Sim2]:
        plot_init_query_points(
            search_space,
            sim,
            objective_data,
            constraint_data,
        )
