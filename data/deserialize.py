import pickle

import tensorflow as tf

def deserialize(raw_bytes):
    serialized = pickle.loads(raw_bytes.numpy())
    genre_code = tf.io.parse_tensor(serialized['genre_code'], tf.int32)
    waveform = tf.io.parse_tensor(serialized['waveform'], tf.int32)
    augmented_waveform = tf.io.parse_tensor(serialized['augmented_waveform'], tf.int32)
    waveform = tf.reshape(waveform, [44100, 1])
    augmented_waveform = tf.reshape(augmented_waveform, [44100, 1])
    return genre_code, waveform, augmented_waveform


def tf_deserialize(raw_bytes):
    genre_code, waveform, augmented_waveform = tf.py_function(
        deserialize,
        [raw_bytes],
        (tf.int32, tf.int32, tf.int32))
    genre_code.set_shape([])
    waveform.set_shape([44100, 1])
    augmented_waveform.set_shape([44100, 1])
    return genre_code, waveform, augmented_waveform


def scale(genre_code, waveform, augmented_waveform):
    scaled_waveform = tf.cast(waveform, tf.float32) / 255.0
    scaled_augmented_waveform = tf.cast(augmented_waveform, tf.float32) / 255.0
    return genre_code, scaled_waveform, scaled_augmented_waveform


def get_deserialized_dataset(list_of_pickled_records, record_byte_size,
                             scale_data=True):
    """Returns a tf.data.Dataset from the list of pickled records

    Args:
      list_of_pickled_records (list of str): List of pickled records
      record_byte_size (int): Byte length for each record
      scale_data (bool): If true, scale waveform data by diving by 255 (max)

    Returns:
      tf.data.Dataset which yields tuples of elements as
        (genre code, waveform, augmented waveform)
    """
    raw_data = tf.data.FixedLengthRecordDataset(list_of_pickled_records,
                                                record_byte_size)
    dataset = raw_data.map(tf_deserialize,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if scale_data:
        dataset = dataset.map(scale,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
