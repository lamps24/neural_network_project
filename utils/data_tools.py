import os
from pathlib import PurePath
from io import StringIO
import warnings
import multiprocessing as mp
from functools import partial

import pandas as pd

import s3_tools

def create_fma_raw_audio_directory(client=None, bucket='fma-dataset',
                                   raw_audio_dir='raw-audio', metadata_dir='fma-metadata'):
  """ Creates a DataFrame that lists raw audio tracks and locations

  Will attach to the tracks metadata and keep relevant features

  Args:
    client (botocore.client.S3): S3 client, if None will create
    bucket (str): Name of S3 bucket
    raw_audio_dir (str): Path to raw audio in bucket
    metadata_dir (str): Path to metadata in bucket
  
  Returns:
    pandas.DataFrame with raw audio tracks and locations in S3 bucket
  """
  if client is None:
    client = s3_tools.get_s3_client()

  # load metadata
  tracks_metadata = s3_tools.load_csv_from_s3(client, bucket,
                                              os.path.join(metadata_dir, 'tracks.csv'),
                                              index_col=0, header=[0, 1])
  features_to_keep = [('track', 'genre_top'),
                      ('track', 'genres'),
                      ('track', 'genres_all'),
                      ('set', 'split'), 
                      ('set', 'subset')]
  tracks_metadata = tracks_metadata[features_to_keep]
  tracks_metadata.columns = tracks_metadata.columns.droplevel()

  # find all mp3 files in directory
  all_files = s3_tools.list_s3_dir(client, bucket, raw_audio_dir)
  mp3_files = [f for f in all_files if PurePath(f).suffix == '.mp3']
  mp3_ids = [int(PurePath(f).stem) for f in mp3_files]
  raw_audio_df = pd.DataFrame.from_dict({'track_id': mp3_ids, 'file': mp3_files})
  raw_audio_df.set_index('track_id', inplace=True)

  # join locations of raw audio to metadata
  raw_audio_directory = raw_audio_df.join(tracks_metadata, how='left')
  assert len(raw_audio_directory) == len(raw_audio_df)

  return raw_audio_directory


def write_fma_raw_audio_directory(bucket='fma-dataset', refresh=False, **kwargs):
  """ Writes out csv of raw audio directory to base of the bucket

  Args:
    bucket (str): Name of S3 bucket
    refresh (bool): If True overwrites any raw audio directory that
      currently exists. If False and a raw audio directory already
      exists, does nothing.
    **kwargs: Keywpord arguments to hand to create_fma_raw_audio_directory

  Returns:
    str with path to raw audio directory csv file on the bucket
  """
  raw_audio_dir_file = 'raw_audio_directory.csv'
  client = s3_tools.get_s3_client()

  if not refresh:
    # check if file already exists
    search_response = client.list_objects(Bucket=bucket, Prefix=raw_audio_dir_file)
    if 'Contents' in search_response:
      warnings.warn('File already exists, aborting creating new directory...')
      return raw_audio_dir_file

  raw_audio_directory = create_fma_raw_audio_directory(client=client, **kwargs)

  resource = s3_tools.get_s3_resource()
  s3_tools.write_csv_to_s3(raw_audio_directory, resource, bucket, raw_audio_dir_file)

  return raw_audio_dir_file


def load_fma_raw_audio_directory(bucket='fma-dataset', create_csv=False, **kwargs):
  """ Loads the csv file of raw audio directory

  If the directory does not exist and create_csv is True, then it creates
  the csv file with default arguments in the given bucket.

  Args:
    bucket (str): Name of S3 bucket
    create_csv (bool): If True and directory does not exists, creates the csv file
    **kwargs: Keyword arguments to hand to write_fma_raw_audio_directory

  Returns:
    pandas.DataFrame with directory to all raw audio tracks in bucket
  """
  raw_audio_dir_file = 'raw_audio_directory.csv'
  
  #check if files exists
  search_response = client.list_objects(Bucket=bucket, Prefix=raw_audio_dir_file)
  if 'Contents' not in search_response and create_csv:
    raw_audio_dir_file = write_fma_raw_audio_directory(bucket=bucket, **kwargs)
  elif 'Contents' not in search_response:
    warnings.warn('Raw audio directory does not exist and create_csv=False, returning None...')
    return None


def create_waveforms_from_raw_audio(bucket='fma-dataset', raw_audio_dir_file='raw_audio_directory.csv',
                                    waveform_dir='waveforms', waveform_dir_file='waveform_directory.csv',
                                    sampling_rate=44100):
  """ CURRENTLY BROKEN! Issues with pickling in multiprocessing...
      Creates a .npy file with the waveform for each mp3 file in raw_audio_dir_file

  Args:
    bucket (str): Name of S3 bucket
    raw_audio_dir_file (str): Key of raw audio directory file in bucket
    waveform_dir (str): Name of directory to save waveforms to
    waveform_dir_file (str): Key to csv file that contains locations of waveforms
    sampling_rate (int): Sampling rate to use in creating waveforms
  """
  client = s3_tools.get_s3_client()
  raw_audio_directory = s3_tools.load_csv_from_s3(client, bucket, raw_audio_dir_file)

  get_waveform = partial(s3_tools.s3_convert_audio_to_npy, client=client, bucket=bucket,
                         npypath=waveform_dir, sr=sampling_rate)

  with mp.Pool(mp.cpu_count()) as p:
    waveform_files = list(p.imap(get_waveform, raw_audio_directory['file'], 25))

  raw_audio_directory['waveform'] = waveform_files
  resource = s3_tools.get_s3_resource()
  s3_tools.write_csv_to_s3(raw_audio_directory, resource, bucket, waveform_dir_file)

if __name__ == '__main__':
  create_waveforms_from_raw_audio()
