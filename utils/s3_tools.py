import os
from io import BytesIO, StringIO
from tempfile import TemporaryFile, NamedTemporaryFile

import boto3
import pandas as pd
import numpy as np

import librosa


def get_s3_client(endpoint_url='https://s3.msi.umn.edu'):
  """ Gets an s3 client for the given endpoint

  To succesfully sign in, must have credentials file ~/.boto
  with access key id and secret access key

  Returns:
    botocore.client.S3 object
  """
  return boto3.client('s3', endpoint_url=endpoint_url)


def get_s3_resource(endpoint_url='https://s3.msi.umn.edu'):
  """ Gets an s3 resource for the given endpoint

  To succesfully sign in, must have credentials file ~/.boto
  with access key id and secret access key

  Returns:
    boto3.resources.factory.s3.ServiceResource
  """
  return boto3.resource('s3', endpoint_url=endpoint_url)


def add_user_to_s3_bucket(client, bucket, uid, permission, name):
  """ Adds user to a S3 bucket by canonical ID
  
  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    uid (str): Canonical user id
    permission (str): Permissions to grant user
    name (str): Display name for user
  """
  bucket_acl = client.get_bucket_acl(Bucket=bucket)
  new_grant = [{'Grantee': {'DisplayName': name,
                            'ID': 'uid=' + str(uid),
                            'Type': 'CanonicalUser'},
                'Permission': permission.to_upper()}]
  bucket_acl['Grants'] += new_grant
  grants_owner = {'Grants': bucket_acl['Grants'], 'Owner': bucket_acl['Owner']}
  client.put_bucket_acl(Bucket=bucket, AccessControlPolicy=grants_owner)


def load_csv_from_s3(client, bucket, fpath, **kwargs):
  """ Loads a csv from an S3 bucket

  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    fpath (str): Path to file in bucket
    **kwargs: Keyword arguments to hand to read_csv

  Returns:
    pandas.DataFrame with data from csv file
  """
  file_obj = client.get_object(Bucket=bucket, Key=fpath)
  df = pd.read_csv(file_obj['Body'], **kwargs)
  return df


def write_csv_to_s3(df, resource, bucket, fpath, **kwargs):
  """ Writes a csv file to an S3 bucket

  Args:
    df (pandas.DataFrame): Dataframe to write to csv
    resource (boto3.resources.factory.s3.ServiceResource): S3 resource
    bucket (str): Name of S3 bucket
    fpath (str): Path to save csv file to
    **kwargs: Keyword arguments to hand to pandas.DataFrame.to_csv()
  """
  csv_io_buffer = StringIO()
  df.to_csv(csv_io_buffer, **kwargs)
  resource.Object(bucket, fpath).put(Body=csv_io_buffer.getvalue())


def load_raw_audio_from_s3(client, bucket, fpath, **kwargs):
  """ Loads raw audio from an mp3 file in an S3 bucket

  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    fpath (str): Path to raw audio file in bucket
    **kwargs: Additional arguments to hand to librosa.load

  Returns:
    numpy.array with waveform of raw audio
  """
  with NamedTemporaryFile(suffix='.mp3') as tempf:
    client.download_fileobj(bucket, fpath, tempf)
    waveform = librosa.load(tempf.name, **kwargs)
  return waveform


def s3_convert_audio_to_npy(client, bucket, fpath, npypath=None, **kwargs):
  """ Converts raw audio in an S3 bucket to .npy file

  Saves the .npy waveform of the raw audio in the same
  directory specified by npypath with the same name as
  the raw audio file (but with the .npy extension)

  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    fpath (str): Path to raw audio file in bucket
    npypath (str): Path to directory to place npy file in
       If not specified, will save npy file in same
       directory as the raw audio file
    **kwargs: Additional arguments to hand to librosa.load

  Returns:
    str of path to npy file in bucket
  """
  if npypath is None:
    npypath = os.path.split(fpath)[0]

  raw_audio_name = os.path.splitext(os.path.split(fpath)[1])[0]
  npyfile_name = os.path.join(npypath, raw_audio_name + '.npy')

  waveform = load_raw_audio_from_s3(client, bucket, fpath, **kwargs)
  with TemporaryFile() as tempf:
    np.save(tempf, waveform)
    tempf.seek(0)
    client.upload_fileobj(tempf, bucket, npyfile_name)
  return npyfile_name


def write_numpy_file_to_s3(numpy_object, client, bucket, fpath):
  """ Saves a npy file to S3

  Args:
    client (botocore.client.S3): S3 client object
    numpy_object (numpy.array): Numpy array to save
    bucket (str): Name of bucket
    fpath (str): Path to numpy file on S3
  """
  with TemporaryFile() as tempf:
    np.save(tempf, numpy_object)
    tempf.seek(0)
    client.upload_fileobj(tempf, bucket, fpath)


def load_numpy_file_from_s3(client, bucket, fpath):
  """ Loads a numpy binary file from a bucket

  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    fpath (str): Path to numpy binary file

  Returns:
    numpy object contained in file
  """
  file_obj = client.get_object(Bucket=bucket, Key=fpath)
  npy_obj = np.load(BytesIO(file_obj['Body'].read()), allow_pickle=True)
  return npy_obj


def list_s3_dir(client, bucket, dirpath):
  """ Returns a list with the absolute path of every file in dirpath (recursively)

  Args:
    client (botocore.client.S3): S3 client object
    bucket (str): Name of bucket
    dirpath (str): Path to directory in bucket

  Returns:
    list of str of keys to each file in dirpath
  """
  paginator = client.get_paginator('list_objects')
  pages = paginator.paginate(Bucket=bucket, Prefix=dirpath)

  keys_in_dir = []
  for page in pages:
    for obj in page['Contents']:
      keys_in_dir += [obj['Key']]

  return keys_in_dir
