import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Conv3D,
    Conv2DTranspose,
    DepthwiseConv2D,
    SeparableConv1D,
    SeparableConv2D,
    AveragePooling1D,
    AveragePooling2D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    MaxPooling1D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    Activation,
)
from tensorflow.python.keras.backend import rnn

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


def test_conv1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    in_ch = 3
    kernel = 32
    ker_w = 3
    ker_h = 3
    ker_z = 3

    model = Sequential(
        Conv1D(kernel, (ker_w,), padding="same", input_shape=(in_w, in_ch))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == ((2 * ker_w * in_ch) + 1) * in_w * kernel

    model = Sequential(
        Conv2D(kernel, (ker_w, ker_h), padding="same", input_shape=(in_w, in_h, in_ch))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == ((2 * ker_w * ker_h * in_ch) + 1) * in_w * in_h * kernel

    model = Sequential(
        Conv3D(
            kernel,
            (ker_w, ker_h, ker_z),
            padding="same",
            input_shape=(in_w, in_h, in_z, in_ch),
        )
    )
    flops = get_flops(model, batch_size=1)
    assert (
        flops == ((2 * ker_w * ker_h * ker_z * in_ch) + 1) * in_w * in_h * in_z * kernel
    )


def test_conv2dtranspose():
    in_w = 32
    in_h = 32
    in_ch = 3
    kernel = 32
    ker_w = 3
    ker_h = 3

    model = Sequential(
        Conv2DTranspose(
            kernel, (ker_w, ker_h), padding="same", input_shape=(in_w, in_h, in_ch)
        )
    )
    flops = get_flops(model, batch_size=1)
    assert flops >= ((2 * ker_w * ker_h * in_ch) + 1) * in_w * in_h * kernel


def test_depthwise_conv2d():
    in_w = 32
    in_h = 32
    in_ch = 3
    ker_w = 3
    ker_h = 3
    model = Sequential(
        DepthwiseConv2D((ker_w, ker_h), padding="same", input_shape=(in_w, in_h, in_ch))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == ((2 * ker_w * ker_h) + 1) * in_w * in_h * in_ch


def test_separable_conv1d2d():
    in_w = 32
    in_h = 32
    in_ch = 3
    kernel = 32
    ker_w = 3
    ker_h = 3

    model = Sequential(
        SeparableConv1D(kernel, (ker_w,), padding="same", input_shape=(in_w, in_ch))
    )
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == 2 * ker_w * in_w * in_ch  # depthwise conv with no bias
        + (2 * in_ch + 1) * in_w * kernel  # pointwise conv
    )

    model = Sequential(
        SeparableConv2D(
            kernel, (ker_w, ker_h), padding="same", input_shape=(in_w, in_h, in_ch)
        )
    )
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == 2 * ker_w * ker_h * in_w * in_h * in_ch  # depthwise conv with no bias
        + (2 * in_ch + 1) * in_w * in_h * kernel  # pointwise conv
    )


def test_averagepooling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32
    pool_w = 2
    pool_h = 2
    pool_z = 2

    model = Sequential(
        AveragePooling1D(pool_size=(pool_w,), input_shape=(in_w, kernel))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * kernel

    model = Sequential(
        AveragePooling2D(pool_size=(pool_w, pool_h), input_shape=(in_w, in_h, kernel))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * kernel


def test_global_averagepooling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32

    model = Sequential(GlobalAveragePooling1D(input_shape=(in_w, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * kernel

    model = Sequential(GlobalAveragePooling2D(input_shape=(in_w, in_h, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * kernel

    model = Sequential(GlobalAveragePooling3D(input_shape=(in_w, in_h, in_z, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * in_z * kernel


def test_maxpooling1d2d3d():
    in_w = 32
    in_h = 32
    kernel = 32
    pool_w = 2
    pool_h = 2

    model = Sequential(MaxPooling1D(pool_size=(pool_w,), input_shape=(in_w, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * kernel

    model = Sequential(
        MaxPooling2D(pool_size=(pool_w, pool_h), input_shape=(in_w, in_h, kernel))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * kernel


def test_softmax():
    kernel = 8
    model = Sequential(Activation("softmax", input_shape=(kernel,)))
    flops = get_flops(model, batch_size=1)
    assert flops == 5 * kernel


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


def test_conv1dtranspose():
    ignore = True
    major, minor, _ = tf.version.VERSION.split(".")
    if int(major) >= 2 and int(minor) >= 3:
        ignore = False
    if ignore:
        return
    from tensorflow.keras.layers import Conv1DTranspose

    in_w = 32

    in_ch = 3
    kernel = 32
    ker_w = 3

    model = Sequential(
        Conv1DTranspose(kernel, (ker_w,), padding="same", input_shape=(in_w, in_ch))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == ((2 * ker_w * in_ch) + 1) * in_w * kernel + 1
