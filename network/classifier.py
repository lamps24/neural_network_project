# import modules
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import tanh, relu, sigmoid

# classification forward pass
# input: 3D tensor of size = [batch_size, length, channels] 
# output: 2D tensor of size = [batch_size, # domains] 
# values of output represent probabilities of belonging to domain
class Classifier(layers.Layer):
    def __init__(self, n_classes, channels=100, kernel_size=1, classifier_layers=3, rate=0.0, **kwargs):
        super().__init__(**kwargs)
  
        # define the layers
        self.n_classes = n_classes
        self.channels = channels
        self.kernel_size = kernel_size
        self.classifier_layers = classifier_layers
        self.rate = rate

        self.initial_conv1d = layers.Conv1D(channels, kernel_size) 
        self.inner_conv1d = layers.Conv1D(channels, kernel_size) 
        self.elu = layers.ELU() 
        self.outer_conv1d = layers.Conv1D(n_classes, kernel_size) 
        self.dropout = layers.Dropout(rate)

    def call(self, encoded_input):
        # execute dropout
        x = self.dropout(encoded_input)

        # execute layers
        for i in range(self.classifier_layers):
          if i == 0:
            x = self.initial_conv1d(x)
          else: 
            x = self.inner_conv1d(x)          
          x = self.elu(x)
        logits = self.outer_conv1d(x)
        
        mean = tf.math.reduce_mean(logits, axis=1)
        return mean


    def get_config(self):
        config = super().get_config()
        config['n_classes'] = self.n_classes
        config['channels'] = self.channels
        config['kernel_size'] = self.kernel_size
        config['classifier_layers'] = self.classifier_layers
        config['rate'] = self.rate

        return config
