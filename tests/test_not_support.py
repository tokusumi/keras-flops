import pytest
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Conv3DTranspose,
    AveragePooling3D,
    MaxPooling3D,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    GlobalMaxPooling3D,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    BatchNormalization,
    LayerNormalization,
)
from keras_flops import get_flops


@pytest.mark.xfail
def test_Embedding():

    sentence = 32
    onehot = 100
    emb = 32

    model = Sequential(Embedding(onehot, emb, input_length=sentence))
    flops = get_flops(model, batch_size=1)
    assert flops > 0, "not supported"


@pytest.mark.xfail
def test_simpleRNN():
    # NOTE: first input dense is only calculated.
    sentence = 32
    emb = 32
    rnn_unit = 3

    model = Sequential(SimpleRNN(rnn_unit, use_bias=False, input_shape=(sentence, emb)))
    flops = get_flops(model, batch_size=1)
    assert (
        flops == 2 * emb * rnn_unit * sentence + 2 * rnn_unit * rnn_unit * sentence
    ), "not supported. first input matmul is only calculated"


@pytest.mark.xfail
def test_lstm():
    sentence = 32
    emb = 32
    rnn_unit = 3

    model = Sequential(LSTM(rnn_unit, use_bias=False, input_shape=(sentence, emb)))
    flops = get_flops(model, batch_size=1)
    assert (
        flops > 2 * emb * rnn_unit * sentence + 2 * rnn_unit * rnn_unit * sentence
    ), "not supported"


@pytest.mark.xfail
def test_gru():
    sentence = 32
    emb = 32
    rnn_unit = 3

    model = Sequential(GRU(rnn_unit, use_bias=False, input_shape=(sentence, emb)))
    flops = get_flops(model, batch_size=1)
    assert (
        flops > 2 * emb * rnn_unit * sentence + 2 * rnn_unit * rnn_unit * sentence
    ), "not supported"


@pytest.mark.xfail
def test_conv1d2d3dtranspose():
    in_w = 32
    in_h = 32
    in_z = 32
    in_ch = 3
    kernel = 32
    ker_w = 3
    ker_h = 3
    ker_z = 3
    model = Sequential(
        Conv3DTranspose(
            kernel,
            (ker_w, ker_h, ker_z),
            padding="same",
            input_shape=(in_w, in_h, in_z, in_ch),
        )
    )
    flops = get_flops(model, batch_size=1)
    assert (
        flops
        == ((2 * ker_w * ker_h * ker_z * in_ch) + 1) * in_w * in_h * in_z * kernel + 3
    )


@pytest.mark.xfail
def test_averagepooling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32
    pool_w = 2
    pool_h = 2
    pool_z = 2

    model = Sequential(
        AveragePooling3D(
            pool_size=(pool_w, pool_h, pool_z), input_shape=(in_w, in_h, in_z, kernel)
        )
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * in_z * kernel


@pytest.mark.xfail
def test_maxpooling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32
    pool_w = 2
    pool_h = 2
    pool_z = 2

    model = Sequential(
        MaxPooling3D(
            pool_size=(pool_w, pool_h, pool_z), input_shape=(in_w, in_h, in_z, kernel)
        )
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * in_z * kernel


@pytest.mark.xfail
def test_global_maxpooling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32

    model = Sequential(GlobalMaxPooling1D(input_shape=(in_w, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * kernel

    model = Sequential(GlobalMaxPooling2D(input_shape=(in_w, in_h, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * kernel

    model = Sequential(GlobalMaxPooling3D(input_shape=(in_w, in_h, in_z, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * in_z * kernel


@pytest.mark.xfail
def test_upsampling1d2d3d():
    in_w = 32
    in_h = 32
    in_z = 32
    kernel = 32
    up_w = 2
    up_h = 2
    up_z = 2

    model = Sequential(UpSampling1D(size=up_w, input_shape=(in_w, kernel)))
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * kernel

    model = Sequential(
        UpSampling2D(size=(up_w, up_h), input_shape=(in_w, in_h, kernel))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * kernel

    model = Sequential(
        UpSampling3D(size=(up_w, up_h, up_z), input_shape=(in_w, in_h, in_z, kernel))
    )
    flops = get_flops(model, batch_size=1)
    assert flops == in_w * in_h * in_z * kernel


@pytest.mark.xfail
def test_batchnormalization():
    """
    batch normalization in tf uses gen_nn_ops.fused_batch_norm_v3 if input shape are 4D
    """
    in_w = 32
    in_h = 32
    in_ch = 3

    model = Sequential(
        BatchNormalization(
            beta_initializer="ones",
            gamma_initializer="ones",
            input_shape=(in_w, in_h, in_ch),
        )
    )
    flops = get_flops(model, batch_size=1)
    assert (
        flops == 5 * in_ch + in_w * in_h * in_ch
    ), "fused is True, fused_batch_norm_v3 is not supportted"


@pytest.mark.xfail
def test_layernormalization():
    """
    layer normalization is calculated as follows,
    fused_
    1. (2 ops * |var|) inv = rsqrt(var + eps)
    2. (1 ops * |var|) inv *= gamma (scale)
    3. (|x| + |mean| + |var| ops) x' = inv * x + beta (shift) - mean * inv
    , where |var| = |mean| = 1 in default
    Thus, 5 channel size + input element size.

    Use nn.fused_batch_norm (gen_nn_ops.fused_batch_norm_v3) for layer normalization, above calculation,
    but gen_nn_ops.fused_batch_norm_v3 is not registered yet, so can not evaluate corrent FLOPs.
    squeezed_shape (ndim ops), scale (|x| ops) and shift (not float ops) is calculated.
    """
    in_w = 32
    in_h = 32
    in_ch = 3

    input_shape = (in_w, in_ch)
    model = Sequential(
        LayerNormalization(scale=False, center=False, input_shape=input_shape,)
    )
    flops = get_flops(model, batch_size=1)
    assert flops == len(input_shape) + 1

    input_shape = (in_w, in_h, in_ch)
    model = Sequential(
        LayerNormalization(scale=False, center=False, input_shape=input_shape,)
    )
    flops = get_flops(model, batch_size=1)
    assert flops == len(input_shape) + 1

    input_shape = (in_w, in_h, in_ch)
    model = Sequential(
        LayerNormalization(
            beta_initializer="ones", gamma_initializer="ones", input_shape=input_shape,
        )
    )
    flops = get_flops(model, batch_size=1)
    assert flops == len(input_shape) + 1 + in_w * in_h * in_ch, "fused is True"

    assert flops == len(input_shape) + 1 + 5 * in_ch + in_w * in_h * in_ch
