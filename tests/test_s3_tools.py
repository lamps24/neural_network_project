import os
import sys
sys.path.append('..')
os.environ['PATH'] += ':~/software/ffmpeg/bin/'

from utils import s3_tools
import unittest

import pandas as pd
import numpy as np
import boto3

import librosa

class TestS3Tools(unittest.TestCase):
  
  def setUp(self):
    self.client = s3_tools.get_s3_client()
    self.bucket = 'fma-dataset'
    test_df = {'a': [0.4, 1, 2, 3, 4],
               'b': [5.6, 1.4, 0, 4, 9.1],
               'c': [0, 0, 0.01, 0.02, 0.01]}
    self.test_df = pd.DataFrame(test_df)
    self.test_df_file = '/home/csci5980/piehl008/csci5980/tests/test_df.csv'
    self.test_df.to_csv(self.test_df_file, index=False)
    self.s3_raw_audio_file = 'raw-audio/000/000182.mp3'
    with open(self.test_df_file, 'rb') as testf:
      self.client.upload_fileobj(testf, self.bucket, 'test_df.csv')
    
    self.mp3_f_local = '/home/csci5980/piehl008/csci5980/tests/000182.mp3'
    self.waveform, self.sr = librosa.load(self.mp3_f_local, sr=44100)

  def test_load_csv(self):
    test_s3_df = s3_tools.load_csv(self.client, self.bucket, 'test_df.csv')
    self.assertTrue(test_s3_df.equals(self.test_df))

  def test_load_raw_audio(self):
    test_wv, sr = s3_tools.load_raw_audio(self.client, self.bucket, self.s3_raw_audio_file, sr=self.sr)
    self.assertTrue(np.array_equal(test_wv, self.waveform))
    self.assertTrue(sr == self.sr)

  def test_convert_audio_to_npy_1(self):
    npy_name = s3_tools.convert_audio_to_npy(self.client, self.bucket, self.s3_raw_audio_file, sr=self.sr)
    ra_list = [obj['Key'] for obj in self.client.list_objects(Bucket=self.bucket, Prefix=os.path.split(npy_name)[0])['Contents']]
    self.assertTrue(npy_name in ra_list)

  def test_load_numpy_file_1(self):
    waveform, sr = s3_tools.load_numpy_file(self.client, self.bucket, 'raw-audio/000/000182.npy')
    self.assertTrue(np.array_equal(waveform, self.waveform))
    self.assertTrue(sr == self.sr)

  def test_convert_audio_to_npy_2(self):
    npy_name = s3_tools.convert_audio_to_npy(self.client, self.bucket, self.s3_raw_audio_file, npypath='raw-audio/npy/000/', sr=self.sr)
    ra_list = [obj['Key'] for obj in self.client.list_objects(Bucket=self.bucket, Prefix=os.path.split(npy_name)[0])['Contents']]
    self.assertTrue(npy_name in ra_list)

  def tearDown(self):
    os.remove(self.test_df_file)
    resource = boto3.resource('s3', endpoint_url='https://s3.msi.umn.edu')
    test_df_obj = resource.Object(self.bucket, self.test_df_file)
    test_df_obj.delete()


def main():
  unittest.main()


if __name__ == '__main__':
  main()
