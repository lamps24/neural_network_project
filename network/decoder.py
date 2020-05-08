import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import tanh, relu, sigmoid


class Wavenet(layers.Layer):
  """Keras Layer subclass that implements a Wavenet-like layer
  
  Implements a Wavenet layer with residual and skip connectsion,
  as well as gated activation units


   |--------------------------------------------------------------------|
   |                                                   ------------     v
   |                              ------------        | Resid conv |--> + --|
   |                        |--->| tanh filter |--|    ------------         |
   |   -------------        |     ------------    v         |               |
  --->| Dilated Conv |-> + -|                     X---------|               --> resid
       -------------     ^  |     -------------   ^         |
                         |  |--->| sigmoid gate |-|         |
                         |        -------------        -----------
                         |                            | Skip conv |-----------> skip
                 ------------------                     -----------
               | Conditional Conv |
                ------------------ 
                         ^ 
                         |
                 conditional input

  In particular, the dilated convolution doubles the size of the
  residual models, and then splits the output into the filter
  and gate activations
  """


  def __init__(self, residual_channels, skip_channels, kernel_size=2,
               dilation=1):
    """Initializes Wavenet layer

    Args:
      residual_channels (int): Number of channels in residual convolution
      skip_channels (int): Number of channels in skip convolution
      kernel_size (int): Kernel size for the dilated convolution
      dilation (int): Dilation rate for the dilated convolution
    """
    super().__init__()
    self.dilated_conv = layers.Conv1D(2 * residual_channels, kernel_size,
                                      padding='causal', dilation_rate=dilation)
    self.skip_conv = layers.Conv1D(skip_channels, 1)
    self.resid_conv = layers.Conv1D(residual_channels, 1)
    self.conditional_conv = layers.Conv1D(2 * residual_channels, 1, padding='same')


  def call(self, inputs, condition):
    """Computes a forward pass of the Wavenet layer

    Args:
      inputs (tf.Tensor): input data, shape=(batch size, sample length, 1)
      condition (tf.Tensor): upsampled conditional input from encoder
        should have shape=(batch size, sample length, encoding channels)
    
    Returns:
      (resid, skip) tuple of tf.Tensor objects
      resid has shape (batch size, sample length, residual_channels)
      skip has shape (batch size, sample length, skip_channels)
    """
    x = self.dilated_conv(inputs)
    z = self.conditional_conv(condition)
    x = x + z
    gate, output = tf.split(x, 2, axis=2)
    x = tanh(output) * sigmoid(gate)
    
    skip = self.skip_conv(x)
    resid = self.resid_conv(x) + inputs

    return resid, skip


class Decoder(layers.Layer):
  """Keras layer sublass that implements the UMT decoder

  Implements the decoder network that transforms the mu-transformed
  audio input and conditional input from the encoder

  The architecture of the network is sketched below


                     input
                       |
                       v
                 -------------
                | Causal Conv |
                 -------------                   -----------
                       |----------------------->| Skip Conv |---------------------|
                       |                         -----------                      |
                       |       ---------                        ----------        |
                       |      | Block of |- residual -> ... -> | Block of |-->    |
                       |----->| Wavenet  |                     | Wavenet  |       |
                              | Layers   |--|                  | Layers   |--|    |
                               ----------   |                   ----------   |    |
                                ^  |   |    |                    ^  |   |    |    |
  condition -> upsampling-------|--------------------------------|  |   |    |    |
                          |        |   |    |                       |   |    |    |
                          |        v   v    v                       v   v    v    v
                          |        --> + -- + --------------------> + - + -- + -> +
                          |        Wavenet layer skips                            |
                          |                                                       v
                          |                                                     ReLU
                          |                                                       |
                          |                                                       v
                          |                                               ------------
                          |                                              | Final Conv |
                          |                                               ------------
                          |                                                       |
                          |          ----------------                             v
                          |-------->| Condition Conv |--------------------------> +
                                     ----------------                             |
                                                                                  v
                                                                                ReLU
                                                                                  |
                                                                                  v
                                                                         ------------
                                                                        | Class Conv |
                                                                         ------------
                                                                                  |
                                                                                  v
                                                                               output

In particular, within each Wavenet block, each successive layer will have twice the
dilation factor of the previous, starting with a dilation of 1

"""


  def __init__(self, wavenet_blocks, wavenet_layers, residual_channels,
               skip_channels, kernel_size, output_classes=256, **kwargs):
    """Initializes a Decoder network layer

    Args:
      wavenet_blocks (int): Number of blocks of wavenet layers
      wavenet_layers (int): Number of wavenet layers per block
      residual_channels (int): Number of residual channels in each Wavenet block
      skip_channels (int): Number of skip channels in each Wavenet block
      kernel_size (int): Kernel size for Wavenet layers and causal conv layer
      output_classes (int): Number of output classes, optional
   """
    super().__init__(**kwargs)


    self.wavenet_blocks = wavenet_blocks
    self.wavenet_layers = wavenet_layers
    self.residual_channels = residual_channels
    self.skip_channels = skip_channels
    self.kernel_size = kernel_size
    self.output_classes = output_classes

    self.dilated_layers = []

    for _ in range(self.wavenet_blocks):
      for i in range(self.wavenet_layers):
        self.dilated_layers.append(Wavenet(self.residual_channels, self.skip_channels,
                                           kernel_size=self.kernel_size,
                                           dilation=2**i))

    self.causal_conv = layers.Conv1D(self.residual_channels, self.kernel_size,
                                     padding='causal')
    self.condition_conv = layers.Conv1D(self.skip_channels, 1)
    self.skip_conv = layers.Conv1D(self.skip_channels, 1)

    self.final_conv = layers.Conv1D(self.skip_channels, 1)
    self.class_conv = layers.Conv1D(self.output_classes, 1)


  def get_config(self):
    """Returns model configuration
    """
    config = super().get_config()
    config['wavenet_blocks'] = self.wavenet_blocks
    config['wavenet_layers'] = self.wavenet_layers
    config['residual_channels'] = self.residual_channels
    config['skip_channels'] = self.skip_channels
    config['kernel_size'] = self.kernel_size
    config['output_classes'] = self.output_classes

    return config

  def call(self, inputs, condition):
    """Computes a forward pass of the decoder

    Args:
      inputs (tf.Tensor): input data, shape=(batch size, sample length, 1)
      condition (tf.Tensor): conditioning data from encoder,
        should be shape (batch size, encoding length, encoding channels)
        This tensor will be upsampled so that the second dimensions (lengths)
        between the input and conditions match
    """
    c = self.upsample(inputs, condition)

    resid = self.causal_conv(inputs)
    skip = self.skip_conv(resid)

    for wavenet_layer in self.dilated_layers:
      resid, s = wavenet_layer(resid, c)
      skip += s

    skip = relu(skip)
    skip = self.final_conv(skip)
    skip = skip + self.condition_conv(c)
    skip = relu(skip)
    out = self.class_conv(skip)

    return out


  def generate(self, condition):
    """Generates decoder output during audio generation

    Feeds the output of the model into itself for generation,
    unlike during training which uses "teacher forcing" and
    ignores the generated output

    UNIMPLEMENTED!
    """
    return None


  @staticmethod
  def upsample(x, c):
    """Upsamples tensor c to have the same second dimension as x

    Args:
      x (tf.Tensor): Input tensor of desired size
      c (tf.Tensor): Conditional tensor to upsample

    Returns:
      tf.Tensor of c upsamples s.t. the 2nd dimension matches
        that of x
    """
    length = x.shape[1]
    encoding_length = c.shape[1]
    upsampling = layers.UpSampling1D(length / encoding_length)
    return upsampling(c)

