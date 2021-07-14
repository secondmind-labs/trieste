# ### Active Learning

# +
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1793)
tf.random.set_seed(1793)
# -

# ### The Problem
#
# active learning is bla bla, in this notebook we will bla bla using bla bla


# +
from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

search_space = Box([0, 0], [1, 1])

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# +
import trieste

observer = trieste.utils.objectives.mk_observer(branin)

num_initial_points = 10
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


# +
import gpflow
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import PredictiveVariance

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return {
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
    }

model = build_model(initial_data)
# -

# ### Active Learning using Predictive Variance

acq = PredictiveVariance()
rule = EfficientGlobalOptimization(builder=acq, optimizer=generate_continuous_optimizer(sigmoid=True))
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer,search_space)

# +
import copy
dataset = copy.deepcopy(initial_data)
model_evolution= []
model_temp = model
bo_iter = 5

#optimize bo once at iteration for capturing model 
for i in range(bo_iter):
    result = bo.optimize(1, dataset, model_temp, rule)
    dataset = result.try_get_final_dataset()
    model_temp = copy.deepcopy(result.try_get_final_model())
    model_evolution.append(model_temp)

# +
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

# +
from util.plotting import plot_bo_points, plot_function_2d

_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)

from util.plotting_plotly import add_bo_points_plotly

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=500, width=500)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
)
fig.show()

# +


for i in range(bo_iter):
    
    def pred_var(x):
        _, var = model_evolution[i].model.predict_f(x)
        return var

    _, ax = plot_function_2d(
        pred_var, search_space.lower, search_space.upper, grid_density=20, contour=True, 
        colorbar=True,     
        figsize=(10, 6),
        title=["Variance contour with queried points at iter:"+str(i+1)],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    plot_bo_points(query_points[:num_initial_points+i+1], ax[0, 0], num_initial_points)


# +
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

for i in range(5):
    mean_pred, var_pred = model_evolution[i].model.predict_f(np.linspace(0.5,2.5,200)[:,None])
    x = np.linspace(0.5,2.5,200)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.linspace(0.5,2.5,200)[:,None], gramacy_lee(np.linspace(0.5,2.5,200)[:,None])[:,0], '-', label="True function")
    ax.plot(x, mean_pred[:,0], '--', alpha=0.5, label="mean prediction")
    ax.scatter(query_points[:num_initial_points,0], observations[:num_initial_points,0],marker="x", color="black", label= "initial points")
    ax.scatter(query_points[num_initial_points:num_initial_points+i+1,0], observations[num_initial_points:num_initial_points+i+1,0], color="orange", label= "queried points")
    ax.fill_between(x, mean_pred[:,0] - 2*np.sqrt(var_pred[:,0]), mean_pred[:,0] + 2*np.sqrt(var_pred[:,0]), alpha=0.2)
    #ax.plot(x, mean_pred[:,0], 'o', color='tab:brown')
    ax.set_title("Iteration "+str(i+1))
    ax.legend()
    plt.savefig("Iteration "+str(i+1)+".jpg")
    plt.show()
    ax.clear
# -

print(model_evolution[14].model.data)

# +


"""
TODO:
plot uncertainty
better color and marker
"""
t = np.linspace(0,2*np.pi)
x = np.sin(t)

fig, ax = plt.subplots(figsize=(10,5))
ax.axis([0.5,2.5,-1,5])
ax.plot(true_data[:,0],true_data[:,1])
scat = ax.scatter([],[])

def animate(i):
    i = i+num_initial_points
    data = np.hstack((query_points[:i], observations[:i]))
    scat.set_offsets(data)

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(query_points)-num_initial_points)
plt.close()
from IPython.display import HTML
HTML(ani.to_jshtml())


# +
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

X = np.linspace(0,3.428, num=250)
Y = np.sin(X*3)

fig = plt.figure(figsize=(13,5), dpi=80)
ax = plt.axes(xlim=(0, 3.428), ylim=(-1,1))
line, = ax.plot([], [], lw=5)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = X[0:(i-1)]+i*0.2
    y = Y[0:(i-1)]-i*0.2
    line.set_data(x,y)
    p = plt.fill_between(x, y, 0, facecolor = 'C0', alpha = 0.2)
    return line, p,

anim = animation.FuncAnimation(fig,animate, init_func=init, 
                               frames = 250, interval=20, blit=True)


HTML(anim.to_jshtml())
# -

# ### Batch Active Learning using Predictive Variance

query_points[:,0]


