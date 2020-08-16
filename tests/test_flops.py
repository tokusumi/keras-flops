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
    BatchNormalization,
    AdditiveAttention,
    Attention,
    Dense,
    Flatten,
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


def test_subclass():
    class SubClass(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = Dense(10)
            self.dense2 = Dense(3)

        def call(self, x):
            x = self.dense1(x)
            return self.dense2(x)

    inp = Input((30,))
    x = SubClass()(inp)
    model = Model(inp, x)
    flops = get_flops(model, 1)
    assert flops == (2 * 30 + 1) * 10 + (2 * 10 + 1) * 3


def test_multi_input():
    inputs = [Input((5,)), Input(5,)]
    out = tf.keras.layers.Multiply()(inputs)
    model = Model(inputs, out)
    flops = get_flops(model, 1)
    assert flops == 5


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


def test_batchnormalization():
    """
    batch normalization is calculated as follows,
    1. (2 ops * |var|) inv = rsqrt(var + eps)
    2. (1 ops * |var|) inv *= gamma (scale)
    3. (|x| + |mean| + |var| ops) x' = inv * x + beta (shift) - mean * inv
    , where |var| = |mean| = channel size in default
    Thus, 5 * channel size + input element size.

    NOTE: support only fused=False
    Use gen_nn_ops.fused_batch_norm_v3 but this is not registered yet and calculated as zero. 
    """
    in_w = 32
    in_h = 32
    in_ch = 3

    model = Sequential(
        BatchNormalization(
            beta_initializer="ones",
            gamma_initializer="ones",
            input_shape=(in_w, in_ch),
        )
    )
    flops = get_flops(model, batch_size=1)
    assert flops == 5 * in_ch + in_w * in_ch, "fused is False"


def test_additive_attention():
    """
    Bahdanau-style attention. query (batch, Tq, dim), key (batch, Tv, dim) and value (batch, Tv, dim) are inputs.
    following computations is processed.
    1. reshape query as shape [batch, Tq, 1, dim] and value as shape [batch, 1, Tv, dim]
    2. broadcasting multiply between both of above as output shape [batch, Tq, Tv, dim]
    3. reduce_sum above with dim axis as output shape [batch, Tq, Tv]
    4. softmax of above
    5. MatMul between 4. and value as output shape [batch, Tq, dim]
    """
    Tq = 10
    Tv = 10
    dim = 16
    q_shape = (Tq, dim)
    k_shape = (Tv, dim)
    v_shape = (Tv, dim)
    q = Input(q_shape)
    k = Input(k_shape)
    v = Input(v_shape)
    x = AdditiveAttention()([q, k, v])
    model = Model([q, k, v], x)
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == Tq * Tv * dim  # No.2 (multiply)
        + Tq * Tv * (dim - 1)  # No.3 (reduce_sum)
        + 5 * Tq * Tv  # No.4 (softmax)
        + 2 * Tv * Tq * dim  # No.5 (MatMul)
    )


def test_attention():
    """
    Luong-style attention. query (batch, Tq, dim), key (batch, Tv, dim) and value (batch, Tv, dim) are inputs.
    following computations is processed.
    1. query-key dot-product as output shape [batch, Tq, Tv]
    2. softmax of above
    3. MatMul between 2. and value as output shape [batch, Tq, dim]
    """
    Tq = 10
    Tv = 10
    dim = 16
    q_shape = (Tq, dim)
    k_shape = (Tv, dim)
    v_shape = (Tv, dim)
    q = Input(q_shape)
    k = Input(k_shape)
    v = Input(v_shape)
    x = Attention()([q, k, v])
    model = Model([q, k, v], x)
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == 2 * Tq * Tv * dim  # No.1 (dot-product (MatMul))
        + 5 * Tq * Tv  # No.2 (softmax)
        + 2 * Tv * Tq * dim  # No.3 (MatMul)
    )
