"""This module runs the training loop for a UMT network

It trains one encoder and one classifier, and an individual
decoder model for each domain.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy

from data.deserialize import get_deserialized_dataset
from network.encoder import Encoder
from network.decoder import Decoder
from network.classifier import Classifier


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--domain_lambda', help='Lambda loss parameter',
                    default=0.01, type=float)
PARSER.add_argument('--training_files_0', required=True,
                    help=('Path to file with list of records for domain 0 on S3, must have one '\
                          'file per line with no other text'))
PARSER.add_argument('--training_files_1', required=True, help='See previous argument')
PARSER.add_argument('--training_files_2', required=True, help='See previous argument')
PARSER.add_argument('--training_record_byte_size', type=int, required=True,
                    help='Byte length for each record in the training dataset')
PARSER.add_argument('--max_epochs', help='Max number of training epochs',
                    default=1000, type=int)
PARSER.add_argument('--batch_size', help='Training batch size',
                    default=200, type=int)
PARSER.add_argument('--sampling_rate', help='Waveform sampling rate',
                    default=44100, type=int)
PARSER.add_argument('--random_seed', help='Random seed for generating training data',
                    default=5980, type=int)
PARSER.add_argument('--n_classes', help='Number of decoding domains',
                    type=int, required=True)
PARSER.add_argument('--scale_data', action='store_true',
                    help='If handed will scale waveform data by diving by 255')
PARSER.add_argument('--shuffle_buffer_size', type=int,
                    help='Buffer size for dataset shuffler')

# encoder arguments
PARSER.add_argument('--encoder_blocks', help='Number of blocks in encoder layer',
                    default=3, type=int)
PARSER.add_argument('--encoder_layers', help='Number of layers in each encoding block',
                    default=10, type=int)
PARSER.add_argument('--encoder_channels', help='Number of encoding channels',
                    default=128, type=int)
PARSER.add_argument('--encoder_kernel_size', help='Encoder kernel size',
                    default=3, type=int)
PARSER.add_argument('--encoder_pool', help='Encoder pooling size',
                    default=450, type=int)


# classifier arguments
PARSER.add_argument('--classifier_channels', help='Number of output classifier channels',
                    default=100, type=int)
PARSER.add_argument('--classifier_kernel_size', help='Classifier kernel size',
                    default=1, type=int)
PARSER.add_argument('--classifier_layers', help='Number of classification layers',
                    default=3, type=int)
PARSER.add_argument('--classifier_dropout_rate', help='Classifier dropout rate',
                    default=0.0, type=float)


# decoder arguments
PARSER.add_argument('--decoder_blocks', help='Number of WaveNet blocks in decoder',
                    default=4, type=int)
PARSER.add_argument('--decoder_layers', help='Number of WaveNet layers per block in decoder',
                    default=10, type=int)
PARSER.add_argument('--decoder_residual_channels', help='Number of residual channels in decoder',
                    default=128, type=int)
PARSER.add_argument('--decoder_skip_channels', help='Number of skip channels in decoder',
                    default=128, type=int)
PARSER.add_argument('--decoder_kernel_size', help='Decoder kernel size',
                    default=2, type=int)


def main():
  """Runs the main training loop

  Creates tensorboard visualizations and saves models after each epoch
  """
  args = PARSER.parse_args()

  start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

  print('Started training with arguments {}'.format(sys.argv))

  np.random.seed(args.random_seed)

  training_files_0 = []
  with open(args.training_files_0, 'r') as training_file_reader_0:
    training_files_0 = training_file_reader_0.readlines()
    training_files_0 = [training_file.strip() for training_file in training_files_0]

  training_files_1 = []
  with open(args.training_files_1, 'r') as training_file_reader_1:
    training_files_1 = training_file_reader_1.readlines()
    training_files_1 = [training_file.strip() for training_file in training_files_1]

  training_files_2 = []
  with open(args.training_files_2, 'r') as training_file_reader_2:
    training_files_2 = training_file_reader_2.readlines()
    training_files_2 = [training_file.strip() for training_file in training_files_2]


  training_dataset_0 = get_deserialized_dataset(training_files_0,
                                                args.training_record_byte_size,
                                                scale_data=args.scale_data)
  training_dataset_0 = training_dataset_0.shuffle(buffer_size=args.shuffle_buffer_size,
                                                  seed=args.random_seed)
  training_dataset_0 = training_dataset_0.batch(args.batch_size)


  training_dataset_1 = get_deserialized_dataset(training_files_1,
                                                args.training_record_byte_size,
                                                scale_data=args.scale_data)
  training_dataset_1 = training_dataset_1.shuffle(buffer_size=args.shuffle_buffer_size,
                                                  seed=args.random_seed)
  training_dataset_1 = training_dataset_1.batch(args.batch_size)


  training_dataset_2 = get_deserialized_dataset(training_files_2,
                                                args.training_record_byte_size,
                                                scale_data=args.scale_data)
  training_dataset_2 = training_dataset_2.shuffle(buffer_size=args.shuffle_buffer_size,
                                                  seed=args.random_seed)
  training_dataset_2 = training_dataset_2.batch(args.batch_size)

  waveform_inputs = Input(shape=(44100, 1), name='waveform_inputs')
  encoded_data = Encoder(args.encoder_blocks, args.encoder_layers,
                         args.encoder_channels, args.encoder_kernel_size,
                         args.encoder_pool, name='encoder')(waveform_inputs)
  classified_data = Classifier(args.n_classes, channels=args.classifier_channels,
                               kernel_size=args.classifier_kernel_size,
                               classifier_layers=args.classifier_layers,
                               rate=args.classifier_dropout_rate,
                               name='classifier')(encoded_data)

  classifier_model = Model(inputs=waveform_inputs, outputs=classified_data)

  def classifier_loss(target_genres, pred_logits):
    return sparse_categorical_crossentropy(target_genres, pred_logits,
                                           from_logits=True)

  classifier_optimizer = tf.keras.optimizers.Adam()
  classifier_loss_history = []

  def classifier_train_step(waveform_list, genres_list):
    """Performs a step of the classifier model
    arguments will be lists of tensors, with
    each element being from a different genre
    """
    waveforms = tf.concat(waveform_list, 0)
    genres = tf.concat(genres_list, 0)

    with tf.GradientTape() as tape:
      logits = classifier_model(waveforms, training=True)
      loss_value = classifier_loss(genres, logits)

    classifier_loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, classifier_model.trainable_variables)
    classifier_optimizer.apply_gradients(zip(grads, classifier_model.trainable_variables))


  def transformer_loss(target_waveform, pred_waveform, target_genres, pred_genres):
    waveform_loss = sparse_categorical_crossentropy(target_waveform, pred_waveform,
                                                    from_logits=True)
    genre_loss = sparse_categorical_crossentropy(target_genres, pred_genres,
                                                 from_logits=True)

    return tf.reduce_sum(waveform_loss, axis=-1) - 0.01 * 44100 * genre_loss

  transformed_0_data = Decoder(args.decoder_blocks, args.decoder_layers,
                               args.decoder_residual_channels,
                               args.decoder_skip_channels, args.decoder_kernel_size,
                               name='decoder_0')(waveform_inputs, encoded_data)
  transformer_0_model = Model(inputs=waveform_inputs,
                              outputs=[transformed_0_data, classified_data])
  transformer_0_optimizer = tf.keras.optimizers.Adam()
  transformer_0_loss_history = []

  def transformer_0_train_step(augmented_waveforms, waveforms, genres):
    """Performs a step of a transformer model
    """
    with tf.GradientTape() as tape:
      waveform_logits, genre_logits = transformer_0_model(augmented_waveforms,
                                                          training=True)
      loss_value = transformer_loss(waveforms, waveform_logits,
                                    genres, genre_logits)

    transformer_0_loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, transformer_0_model.trainable_variables)
    transformer_0_optimizer.apply_gradients(zip(grads, transformer_0_model.trainable_variables))


  transformed_1_data = Decoder(args.decoder_blocks, args.decoder_layers,
                               args.decoder_residual_channels,
                               args.decoder_skip_channels, args.decoder_kernel_size,
                               name='decoder_1')(waveform_inputs, encoded_data)
  transformer_1_model = Model(inputs=waveform_inputs,
                              outputs=[transformed_1_data, classified_data])
  transformer_1_optimizer = tf.keras.optimizers.Adam()
  transformer_1_loss_history = []

  def transformer_1_train_step(augmented_waveforms, waveforms, genres):
    """Performs a step of a transformer model
    """
    with tf.GradientTape() as tape:
      waveform_logits, genre_logits = transformer_1_model(augmented_waveforms, training=True)
      loss_value = transformer_loss(waveforms, waveform_logits,
                                    genres, genre_logits)

    transformer_1_loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, transformer_1_model.trainable_variables)
    transformer_1_optimizer.apply_gradients(zip(grads, transformer_1_model.trainable_variables))


  transformed_2_data = Decoder(args.decoder_blocks, args.decoder_layers,
                               args.decoder_residual_channels,
                               args.decoder_skip_channels, args.decoder_kernel_size,
                               name='decoder_2')(waveform_inputs, encoded_data)
  transformer_2_model = Model(inputs=waveform_inputs,
                              outputs=[transformed_2_data, classified_data])
  transformer_2_optimizer = tf.keras.optimizers.Adam()
  transformer_2_loss_history = []

  def transformer_2_train_step(augmented_waveforms, waveforms, genres):
    """Performs a step of a transformer model
    """
    with tf.GradientTape() as tape:
      waveform_logits, genre_logits = transformer_2_model(augmented_waveforms, training=True)
      loss_value = transformer_loss(waveforms, waveform_logits,
                                    genres, genre_logits)

    transformer_2_loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, transformer_2_model.trainable_variables)
    transformer_2_optimizer.apply_gradients(zip(grads, transformer_2_model.trainable_variables))



  log_dir = "logs/fit/" + start
  models_dir = 'models/' + start
  os.mkdir(models_dir)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        write_images=False)
  tensorboard_callback.set_model(classifier_model)
  summary_writer = tf.summary.create_file_writer(log_dir + '/train')


  def train(epochs):
    global_steps = 0
    for epoch in range(epochs):
      for (batch, \
           ((genre_code_0, waveform_0, augmented_waveform_0), \
            (genre_code_1, waveform_1, augmented_waveform_1), \
            (genre_code_2, waveform_2, augmented_waveform_2))) \
           in enumerate(zip(training_dataset_0, training_dataset_1, training_dataset_2)):
        print('Epoch {} batch {} fit:'.format(epoch, batch))

        classifier_start = time.time()
        classifier_train_step([augmented_waveform_0, augmented_waveform_1,
                               augmented_waveform_2],
                              [genre_code_0, genre_code_1, genre_code_2])
        classifier_time = time.time() - classifier_start

        print('\tClassifier, fit time {}, loss {}'.format(classifier_time,
                                                          classifier_loss_history[-1]))

        transformer_0_start = time.time()
        transformer_0_train_step(augmented_waveform_0, waveform_0, genre_code_0)
        transformer_0_time = time.time() - transformer_0_start

        print('\tTransformer 0, fit time {}, loss {}'.format(transformer_0_time,
                                                             transformer_0_loss_history[-1]))

        transformer_1_start = time.time()
        transformer_1_train_step(augmented_waveform_1, waveform_1, genre_code_1)
        transformer_1_time = time.time() - transformer_1_start

        print('\tTransformer 1, fit time {}, loss {}'.format(transformer_1_time,
                                                             transformer_1_loss_history[-1]))

        transformer_2_start = time.time()
        transformer_2_train_step(augmented_waveform_2, waveform_2, genre_code_2)
        transformer_2_time = time.time() - transformer_2_start
        print('\tTransformer 2, fit time {}, loss {}'.format(transformer_2_time,
                                                             transformer_2_loss_history[-1]))

        if (batch + 1) % 100 == 0:
          with summary_writer.as_default():
            tf.summary.scalar('classifier loss', classifier_loss_history[-1],
                              step=global_steps)
            tf.summary.scalar('transformer 0 loss', transformer_0_loss_history[-1],
                              step=global_steps)
            tf.summary.scalar('transformer 1 loss', transformer_1_loss_history[-1],
                              step=global_steps)
            tf.summary.scalar('transformer 2 loss', transformer_2_loss_history[-1],
                              step=global_steps)
            tf.summary.flush()
        global_steps += 1

      # save weights every epoch
      print('Saving model weights')
      classifier_model_save_str = '/classifier_weights.{:d}-{:.2f}.h5'.format(
          epoch, classifier_loss_history[-1])
      classifier_model.save_weights(models_dir + classifier_model_save_str)
      print('Finished training epoch {}, epoch losses are:'.format(epoch))
      print('\tClassifier loss = {}'.format(classifier_loss_history[-1]))
      print('\tTransformer 0 loss = {}'.format(transformer_0_loss_history[-1]))
      print('\tTransformer 1 loss = {}'.format(transformer_1_loss_history[-1]))
      print('\tTransformer 2 loss = {}'.format(transformer_2_loss_history[-1]))

      transformer_0_save_str = '/transformer_0_weights.{:d}-{:.2f}.h5'
      transformer_0_save_str = transformer_0_save_str.format(epoch, transformer_0_loss_history[-1])
      transformer_0_model.save_weights(models_dir + transformer_0_save_str)

      transformer_1_save_str = '/transformer_1_weights.{:d}-{:.2f}.h5'
      transformer_1_save_str = transformer_1_save_str.format(epoch, transformer_1_loss_history[-1])
      transformer_1_model.save_weights(models_dir + transformer_1_save_str)

      transformer_2_save_str = '/transformer_2_weights.{:d}-{:.2f}.h5'
      transformer_2_save_str = transformer_2_save_str.format(epoch, transformer_2_loss_history[-1])
      transformer_2_model.save_weights(models_dir + transformer_2_save_str)

      with summary_writer.as_default():
        tf.summary.scalar('classifier loss', classifier_loss_history[-1],
                          step=global_steps)
        tf.summary.scalar('transformer 0 loss', transformer_0_loss_history[-1],
                          step=global_steps)
        tf.summary.scalar('transformer 1 loss', transformer_1_loss_history[-1],
                          step=global_steps)
        tf.summary.scalar('transformer 2 loss', transformer_2_loss_history[-1],
                          step=global_steps)
      tf.summary.flush()

  train(args.max_epochs)

if __name__ == '__main__':
  main()
