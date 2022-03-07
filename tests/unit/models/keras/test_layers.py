from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from trieste.models.keras.layers import DropConnect
from tests.util.misc import ShapeLike, empty_dataset, random_seed

@pytest.fixture(name="x", params=[tf.constant([[5., 3.4, 2.6], [5.4, 3.2, 1.]])])
def _x_fixture(request: Any) -> tf.Tensor:
    return request.param

@pytest.fixture(name="activation", params=[tf.nn.relu])
def _activation_fixture(request: Any):
    return request.param

@pytest.fixture(name="units", params=[3])
def _units_fixture(request: Any):
    return request.param


@pytest.mark.parametrize("layer", [DropConnect(units=3)])
def test_dense_forward(layer:Dense, x, activation, units):
    '''Tests the forward method is working properly within the model without dropout'''

    layer.units=units
    layer.activation=activation

    inputs = Input(shape=(x.shape[-1],))
    outputs = layer(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    dense_outputs = Dense(units=units, activation=activation, weights=model.get_weights())(inputs)
    dense_model = Model(inputs=inputs, outputs=dense_outputs)
    
    assert (tf.equal(model(x), layer(x))).numpy().all(), "Forward pass within a model is not a forward pass for the layer"
    assert (tf.equal(model.predict(x), model(x))).numpy().all(), "Model predict is not the same as a forward pass"
    assert (tf.equal(dense_model(x), model(x))).numpy().all(), "Forward pass calculations are wrong"

# @random_seed
@pytest.mark.parametrize("rate", [0, 1])
@pytest.mark.parametrize("dropout_layer", [DropConnect(units=3)])
def test_fit(dropout_layer, x, units, activation, rate):
    '''Tests that the fit method with dropout is working properly'''
    y = tf.constant([[3., -4., 9.], [5., 1., 6.]])
    
    inputs = Input(shape=x.shape[-1])
    dropout_layer.activation=activation
    dropout_layer.rate=rate
    dropout_layer.units=units
    dense = Dense(units=units, activation=activation)

    drop_model = Model(inputs=inputs, outputs=dropout_layer(inputs))
    dense_model = Model(inputs=inputs, outputs=dense(inputs))
    drop_model.compile(Adam(), MeanAbsoluteError())
    dense_model.compile(Adam(), MeanAbsoluteError())

    bias = drop_model.get_weights()[1]
    weights = tf.zeros(shape=drop_model.get_weights()[0].shape) if rate == 1 \
        else drop_model.get_weights()[0]
    dense.set_weights([weights, bias])

    drop_fit = drop_model.fit(x, y)
    dense_fit = dense_model.fit(x, y)

    npt.assert_approx_equal(drop_fit.history["loss"][0], dense_fit.history["loss"][0], significant=3,
        err_msg=f"Expected {dropout_layer} to drop {rate} variables and get the same fit as an equivalent dense layer")
    


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("drop_layer", [DropConnect(units=1, use_bias=False)])
def test_dropout_rate(rate, drop_layer):
    '''Tests that weights are being dropped out at the write proportion'''
    drop_layer.rate = rate
    x1 = tf.constant ([[1.]])
    sims=1000
    simulations = [np.sum(drop_layer(x1, training=True).numpy() == 0.) for _ in range(sims)]
    
    #Test dropout up to twice the variance
    assert np.abs(np.sum(simulations) - rate * sims) <= 1.5 * rate * (1-rate) * sims, \
        f"Expected to dropout around {rate} of the passes but only dropped {np.sum(simulations)/len(simulations)}"
