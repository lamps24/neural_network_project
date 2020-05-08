import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.activations import tanh, relu, sigmoid

class DilatedResConv(layers.Layer):
  def __init__(self, channels, dilation, kernel_size):
    super().__init__()

    #defaults to stride = 1, use_bias = TRUE
    self.dilated_conv = layers.Conv1D(channels, kernel_size=kernel_size, dilation_rate=dilation,
                                      padding='same')
    self.conv_1x1 = layers.Conv1D(channels, kernel_size = 1,
                                  padding='same')


  def call(self, inputs):
    x = inputs
    x = relu(x)
    x = self.dilated_conv(x)
    x = relu(x) # default will be using the ReLU activation, but can change if necessary
    x = self.conv_1x1(x)

    return inputs + x

class Encoder(layers.Layer):
  def __init__(self, encoder_blocks, encoder_layers, encoder_channels, kernel_size, encoder_pool, **kwargs):
    super().__init__(**kwargs)

    self.encoder_blocks = encoder_blocks
    self.encoder_layers = encoder_layers
    self.encoder_channels = encoder_channels
    self.kernel_size = kernel_size
    self.encoder_pool = encoder_pool

    dilated_layers = []
    for _ in range(self.encoder_blocks):
      for i in range(self.encoder_layers):
        dilation = 2 ** i
        dilated_layers.append(DilatedResConv(channels=self.encoder_channels, kernel_size=self.kernel_size, dilation=dilation))
    self.dilated_convs = Sequential(layers=dilated_layers)
    self.start = layers.Conv1D(self.encoder_channels, self.kernel_size) # default padding is 'valid', for Non-Causal Convulution
    self.conv_1x1 = layers.Conv1D(self.encoder_channels, kernel_size = 1)
    self.pool = layers.AveragePooling1D(self.encoder_pool, padding='same')

  def call(self, inputs):
    x = self.start(inputs)
    x = self.dilated_convs(x)
    x = self.conv_1x1(x)
    x = self.pool(x)

    return x


  def get_config(self):
    config = super().get_config()
    config['encoder_blocks'] = self.encoder_blocks
    config['encoder_layers'] = self.encoder_layers
    config['kernel_size'] = self.kernel_size
    config['encoder_pool'] = self.encoder_pool

    return config
