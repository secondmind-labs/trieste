from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from .architectures import DropoutNetwork
from .layers import DropConnect, MCDropConnect, MCDropout 
from tests.util.misc import ShapeLike, empty_dataset, random_seed

@pytest.mark.parametrize("units", [3])
@pytest.mark.parametrize("activation", [tf.nn.relu])
@pytest.mark.parametrize("layer", [DropConnect(units=3)])
@pytest.mark.parametrize("x", [tf.constant([[5., 3.4, 2.6], [5.4, 3.2, 1.]])])
def test_dense_forward(layer:Dense, x, activation, units):
    '''Tests the forward method is working properly within the model without dropout'''

    layer.units=units
    layer.activation=activation

    inputs = Input(shape=(units,))
    outputs = layer(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    dense_outputs = Dense(units=units, activation=activation, weights=model.get_weights())(inputs)
    dense_model = Model(inputs=inputs, outputs=dense_outputs)
    
    assert (tf.equal(model(x), layer(x))).numpy().all(), "Forward pass within a model is not a forward pass for the layer"
    assert (tf.equal(model.predict(x), model(x))).numpy().all(), "Model predict is not the same as a forward pass"
    assert (tf.equal(dense_model(x), model(x))).numpy().all(), "Forward pass calculations are wrong"

# def test_fit():
#     '''Tests that the fit method with dropout is working properly'''
#     x = tf.constant([[5., 3.4, 2.6], [5.4, 3.2, 1.]])

#     weights = [
#         np.array([
#             [3., 6., -19.],
#             [2., -5., 9.],
#             [1., -1., 1.]
#         ]),
#         np.array([1., 3., 5.])
#     ]

#     y = tf.constant([[3., -4., 9.], [5., 1., 6.]])

#     dc1 = DropConnect(p_dropout = 1, units = 3, activation="relu")
#     dc0 = DropConnect(p_dropout = 0, units = 3, activation = "relu")

#     inputs = Input(shape=(3,))
#     m1 = Model(inputs=inputs, outputs=dc1(inputs))
#     m0 = Model(inputs=inputs, outputs=dc0(inputs))

#     m0.set_weights(weights)
#     m1.set_weights(weights)

#     m0.compile(Adam(), MeanAbsoluteError())
#     m1.compile(Adam(), MeanAbsoluteError())

#     h0 = m0.fit(x, y)
#     h1 = m1.fit(x, y)
    
#     assert h0.history["loss"][0] - 15.3 <= 1e-3, "Fit method not keeping weights with p = 0"
#     assert h1.history["loss"][0] - 3.3333 <= 1e-3, "Fit method not dropping weights with p = 1"


# def test_p_dropout():
#     '''Tests that weights are being dropped out at the write proportion'''
#     p=0.7
#     x1 = tf.constant ([[1.]])
#     dc1 = DropConnect(p_dropout = p, units = 1)
#     simulations = [np.sum(dc1(x1, training=True).numpy() == 0.) for _ in range(10000)]
    
#     assert 6850 <= np.sum(simulations) <=7150, \
#         f"Expected to dropout around {p} of the passes but only dropped {np.sum(simulations)/len(simulations)}"
