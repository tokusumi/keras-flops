# keras-flops

![](https://github.com/tokusumi/keras-flops/workflows/Tests/badge.svg)
[![PyPI version](https://badge.fury.io/py/keras-flops.svg)](https://badge.fury.io/py/keras-flops)

FLOPs calculator for neural network architecture written in tensorflow (tf.keras) v2.2+

This stands on the shoulders of giants, [tf.profiler](https://www.tensorflow.org/api_docs/python/tf/compat/v1/profiler/Profiler). 

## Requirements

* Python 3.6+
* Tensorflow 2.2+

## Installation

Using pip:

```
pip install keras-flops
```

## Example

See colab examples [here](https://github.com/tokusumi/keras-flops/tree/master/notebooks) in details.

```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras_flops import get_flops

# build model
inp = Input((32, 32, 3))
x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(10, activation="softmax")(x)
model = Model(inp, out)

# Calculae FLOPS
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
# >>> FLOPS: 0.0338 G
```

## Support

Support `tf.keras.layers` as follows,

| name | layer | 
| -- | -- |
| Conv | Conv[1D/2D/3D]|
| | Conv[1D/2D]Transpose |
| | DepthwiseConv2D |
| | SeparableConv[1D/2D] |
| Pooling | AveragePooling[1D/2D] |
| | GlobalAveragePooling[1D/2D/3D]|
| | MaxPooling[1D/2D] |
| | GlobalMaxPool[1D/2D/3D] |
| Normalization | BatchNormalization |
| Activation | Softmax |
| Attention | Attention |
| | AdditiveAttention |
| others | Dense |

## Not supported

Not support `tf.keras.layers` as follows. They are calculated as zero or smaller value than correct value.

| name | layer | 
| -- | -- |
| Conv | Conv3DTranspose |
| Pooling | AveragePooling3D |
| | MaxPooling3D |
| | UpSampling[1D/2D/3D] |
| Normalization | LayerNormalization |
| RNN | SimpleRNN |
| | LSTM |
| | GRU |
| others | Embedding |