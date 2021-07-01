import numpy as np
import gpflow
import trieste
import tensorflow as tf

from gspheres.kernels import SphericalMatern, ChordMatern
from gspheres.fundamental_set import num_harmonics
from gspheres.vish import SphericalHarmonicFeatures
from gpflow.models import SVGP

from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly, add_bo_points_plotly, plot_gp_plotly
from trieste.space import Box
from trieste.acquisition.rule import OBJECTIVE

# Set up data
search_space = Box([0, 0], [1, 1])
observer = trieste.utils.objectives.mk_observer(branin, OBJECTIVE)

num_initial_points = 35
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# Add bias dimension
bias = 1.
X = np.concatenate((initial_query_points * 2. - 1., bias * np.ones((num_initial_points, 1))), axis=1)
# Map to hypersphere
X = X / np.linalg.norm(X, axis=1, keepdims=True)

data = (initial_data[OBJECTIVE].query_points.numpy(), initial_data[OBJECTIVE].observations.numpy() )

# Set up model
# max_degree = 5
# kernel = ChordMatern(nu=0.5, dimension=3)
# _ = kernel.eigenvalues(max_degree)


# m = SVGP(
#     kernel=kernel,
#     likelihood=gpflow.likelihoods.Gaussian(variance=1e-5),
#     inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=max_degree),
#     num_data=len(X),
#     whiten=False
# )
# # gpflow.set_trainable(m.kernel, False)
# gpflow.set_trainable(m.likelihood, False)
# # gpflow.set_trainable(m.inducing_variable, False)
#
# opt = gpflow.optimizers.Scipy()
# opt.minimize(
#     m.training_loss_closure(data, compile=False), m.trainable_variables, compile=False
# )
#
# print(f'ELBO: {m.elbo(data).numpy().item()}')

def get_svgp(data):
    # degrees = 3
    # truncation_level = np.sum([num_harmonics(3, d) for d in range(degrees)])
    max_degree = 5
    kernel = ChordMatern(nu=0.5, dimension=3, variance=tf.math.reduce_variance(data[1]))
    _ = kernel.eigenvalues(max_degree)
    model = gpflow.models.SVGP(
        # kernel=SphericalMatern(nu=0.5, truncation_level=truncation_level, dimension=3),
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=1e-5),
        inducing_variable=SphericalHarmonicFeatures(dimension=3, degrees=max_degree),
        num_data=len(data[0]),
        whiten=False,
    )
    gpflow.utilities.set_trainable(model.likelihood, False)

    opt = gpflow.optimizers.Scipy()
    print(model.trainable_variables)
    opt.minimize(model.training_loss_closure(data), model.trainable_variables)
    return model

m = get_svgp(data)

# # Prediction
# def predict_mean_on_original(x):
#     x = np.concatenate((x * 2. - 1., bias * np.ones((x.shape[0], 1))), axis=1)
#     r =  np.linalg.norm(x, axis=1, keepdims=True)
#     x = x / r
#     mu, _ = m.predict_f(x)
#     return mu * r


# fig = plot_function_plotly(predict_mean_on_original, search_space.lower, search_space.upper, grid_density=20)
fig = plot_gp_plotly(m, search_space.lower, search_space.upper, grid_density=20)

fig.update_layout(height=800, width=1000)

fig = add_bo_points_plotly(
    x=initial_data[OBJECTIVE].query_points[:, 0],
    y=initial_data[OBJECTIVE].query_points[:, 1],
    z=initial_data[OBJECTIVE].observations[:, 0],
    num_init=num_initial_points,
    fig=fig,
)
fig.show()


from trieste.models import SparseVariational
from trieste.type import TensorType
from trieste.data import Dataset
from trieste.models.optimizer import Optimizer




# class Vish(SparseVariational):
#
#     def __init__(self, model: SVGP, data: Dataset, bias: float, optimizer: Optimizer = None):
#         super().__init__(optimizer=optimizer, data=data, model=model)
#         self._bias = bias
#
#     def predict(self, query_points: TensorType):
#         r = np.linalg.norm(query_points, axis=1, keepdims=True)
#         mu, v = self.model.predict_f(self.rescale_input(query_points))
#         return mu * r, v * r**2
#
#     def predict_f(self, query_points: TensorType):
#         r = np.linalg.norm(query_points, axis=1, keepdims=True)
#         mu, v = self.model.predict_f(self.rescale_input(query_points))
#         return mu * r, v * r ** 2
#
#     def predict_joint(self, query_points: TensorType):
#         return self.model.predict_f(self.rescale_input(query_points), full_cov=True)
#
#     def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
#         return self.model.predict_f_samples(self.rescale_input(query_points), num_samples)
#
#     def rescale_input(self, x):
#         x = np.concatenate((x * 2. - 1., self._bias * np.ones((x.shape[0], 1))), axis=1)
#         return x / np.linalg.norm(x, axis=1, keepdims=True)


# vm = m  #SVGP(model=m, bias=1, data=initial_data[OBJECTIVE])
#
# from util.plotting_plotly import plot_gp_plotly
#
# fig = plot_gp_plotly(
#     vm,  # type: ignore
#     search_space.lower,
#     search_space.upper,
#     grid_density=30,
# )

# fig = add_bo_points_plotly(
#     x=initial_data[OBJECTIVE].query_points[:, 0],
#     y=initial_data[OBJECTIVE].query_points[:, 1],
#     z=initial_data[OBJECTIVE].observations[:, 0] / 100,
#     num_init=num_initial_points,
#     fig=fig,
# )

# fig.show()