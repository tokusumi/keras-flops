import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Activation,
)

from keras_flops import get_flops


def test_raise():
    try:
        raised = False
        get_flops(None)
    except KeyError:
        raised = True

    assert raised


def test_duplicated_calculation():
    model = Sequential(Dense(5, input_shape=(3,)))
    flops1 = get_flops(model)
    flops2 = get_flops(model)
    flops3 = get_flops(model)
    assert flops1 == flops2 and flops2 == flops3


def test_ignore():
    model = Sequential(
        [Flatten(input_shape=(16, 16)), Activation("relu"), Dropout(0.25),]
    )
    flops = get_flops(model, 1)
    assert flops == 0


def test_dense():
    in_dense = 8
    out_dense = 3

    def build_model1():
        inp = Input((in_dense,))
        out = Dense(out_dense)(inp)
        model = Model(inp, out)
        return model

    model = build_model1()
    flops = get_flops(model, batch_size=1)
    assert flops == 2 * in_dense * out_dense + out_dense

    def build_model2():
        return Sequential(Dense(out_dense, use_bias=False, input_shape=(in_dense,)))

    model = build_model2()
    flops = get_flops(model, batch_size=1)
    assert flops == 2 * in_dense * out_dense


def test_cnn():
    in_w = 32
    in_h = 32
    in_ch = 3
    kernel = 32
    ker_w = 3
    ker_h = 3
    pool_w = 2
    pool_h = 2

    def build_model():
        inp = Input((in_w, in_h, in_ch))
        x = Conv2D(kernel, (ker_w, ker_h), padding="same")(inp)
        x = MaxPooling2D(pool_size=(pool_w, pool_h))(x)
        return Model(inp, x)

    model = build_model()
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == ((2 * ker_w * ker_h * in_ch) + 1) * in_w * in_h * kernel  # Conv2D
        + in_w * in_h * kernel  # max pooling2D
    )
