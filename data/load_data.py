import sys
sys.path += ['../utils']

import numpy as np
import tensorflow as tf
import librosa

from sklearn.preprocessing import LabelEncoder

import s3_tools


def augment_waveform(waveform, sr=44100):
  """ Randomly selects a segment of the waveform to shift pitch
  
  Will randomly select a length between 0.25 and 0.5 and random
  start between 0 and 0.5, and will shift the pitch of the
  waveform between the start*len(waveform) to (start+length)*len(waveform)
  by a random fractional shift between -0.5 and 0.5.

  Args:
    waveform (numpy.ndarry): array containing waveform
    sr (int): sampling rate for the waveform

  Returns:
    numpy.ndarray containing the augmented waveform
      will have the same shape as waveform
  """
  aug_length, aug_start, aug_pitch = np.random.random(3)
  aug_length = aug_length / 4 + 1 / 4
  aug_start = aug_start / 2
  aug_pitch = aug_pitch - 1/2
  aug_end = aug_start + aug_length

  int_start = int(aug_start * waveform.shape[0])
  int_end = int(aug_end * waveform.shape[0])

  waveform_aug = librosa.effects.pitch_shift(waveform[int_start:int_end],
                                             sr, aug_pitch)
  waveform_aug = np.concatenate([waveform[:int_start], waveform_aug, waveform[int_end:]])
  
  return waveform_aug


def mu_transform(waveform, mu=255):
  """ Performs an 8-bit mu-transform on the waveform

  Args:
    waveform (numpy.ndarray): numpy array containing waveform
      All elements must be between -1 and 1
    mu (int): mu parameter for transformation

  Returns:
    (numpy.ndarray) resulting quantized transformed waveform,
      will be of type int
  """
  waveform_q = np.sign(waveform) * np.log(1 + mu * np.abs(waveform)) / np.log(1 + mu)
  waveform_q = (255 * (waveform_q + 1) / 2).astype('int')
  return waveform_q


def train_data_gen(training_csv, sr, bucket, seed=5):
  """ Generator that generates training observations

  Args:
    training_csv (str): key to training csv file on S3 bucket
    sr (int): sampling rate for training data
    bucket (str): name of S3 bucket containing data
    seed (int): random seed for shuffling dataset

  Yields:
    (str, numpy.ndarray, numpy.ndarray) containing
      (top genre label, mu transformed waveform
       mu transformed augmented waveform)
  """
  np.random.seed(seed)

  training_csv = str(training_csv, 'utf-8')
  bucket = str(bucket, 'utf-8')

  s3_client = s3_tools.get_s3_client()
  training_df = s3_tools.load_csv_from_s3(s3_client, bucket, training_csv)
  ids_shuffle = np.random.permutation(training_df.index.values)

  le = LabelEncoder()
  training_df['genre_top_encoded'] = le.fit_transform(training_df['genre_top'].to_numpy())

  for i in ids_shuffle:
    descriptors = training_df.iloc[i]
    top_genre = descriptors['genre_top']
    top_genre_code = descriptors['genre_top_encoded']

    waveform = s3_tools.load_numpy_file_from_s3(s3_client, bucket,
                                                descriptors['waveform_file'])
    waveform_augmented = augment_waveform(waveform, sr)

    waveform = np.expand_dims(mu_transform(waveform), axis=1)
    waveform_augmented = np.expand_dims(mu_transform(waveform_augmented), axis=1)

    yield top_genre, top_genre_code, waveform, waveform_augmented


def get_train_dataset(training_csv, sr=44100, bucket='fma-dataset', seed=5):
  """ Creates tensorflow dataset for training data

  For example, with default settings you can run
  get_train_dataset('sr44100_waveform_training_directory.csv')

  Args:
    training_csv (str): csv file containing list of training files
    sr (int): sampling rate for training data
    bucket (str): name of S3 bucket containing data, optional
    seed (int): random seed for shuffling dataset, optional

  Returns:
    tup with first argument containing number of distinct
      genres in training dataset, second the number of samples
      in the dataset, and third containing
      tensorflow.data.DataSet containing training data
  """
  args = (training_csv, sr, bucket, seed)
  
  s3_client = s3_tools.get_s3_client()
  training_df = s3_tools.load_csv_from_s3(s3_client, bucket, training_csv)
  n_classes = len(np.unique(training_df['genre_top'].to_numpy()))
  n_samples = len(training_df)

  dataset = tf.data.Dataset.from_generator(train_data_gen,
                                           (tf.string, tf.int16, tf.int16, tf.int16),
                                           (tf.TensorShape([]), tf.TensorShape([]),
                                            tf.TensorShape([sr, 1]), tf.TensorShape([sr, 1])),
                                           args=args)
  return n_classes, n_samples, dataset
