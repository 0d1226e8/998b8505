import numpy as np


def _initialize_tf_dataset(file_names,
                           augment,
                           over_sample,
                           shuffle,
                           target_size,
                           normalize,
                           nb_workers=8,
                           batch_size=64,
                           shuffle_buffer_size=3000,
                           input_type="img",
                           nr_epochs=1):
  import tensorflow as tf
  from tensorflow.contrib.data import map_and_batch
  from tensorflow.contrib.data import shuffle_and_repeat

  if not type(target_size) is list:
    target_size = list(target_size)

  with tf.name_scope('input_pipeline'):
    dataset = tf.data.TFRecordDataset(file_names)
    if shuffle:
      dataset = dataset.apply(
          shuffle_and_repeat(shuffle_buffer_size, nr_epochs))

    def _decode_and_augment_image(example_proto):
      keys_to_features = {
          'label': tf.FixedLenFeature([], tf.int64),
          'shape': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
      }
      tfrecord_features = tf.parse_single_example(example_proto,
                                                  keys_to_features)

      image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
      shape = tf.decode_raw(tfrecord_features['shape'], tf.int64)
      if input_type == ".jpeg":
        image = tf.reshape(image, target_size + [3])
      else:
        image = tf.reshape(image, target_size)
      label = tfrecord_features['label']

      if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        degrees = tf.random_uniform((), minval=-180, maxval=180)
        image = tf.contrib.image.rotate(image, degrees)

        width_shift = tf.random_uniform((), minval=0, maxval=0.05)
        height_shift = tf.random_uniform((), minval=0, maxval=0.05)

        horizontal_pad = tf.cast(
            tf.ceil(width_shift * target_size[0]), tf.int32)
        vertical_pad = tf.cast(tf.ceil(height_shift * target_size[1]), tf.int32)

        padding = tf.stack([
            horizontal_pad, horizontal_pad, vertical_pad, vertical_pad,
            tf.constant(0),
            tf.constant(0)
        ])
        padding = tf.reshape(padding, (3, 2))

        image = tf.pad(image, padding)
        image = tf.random_crop(image, target_size + [3])

        zoom = tf.random_uniform((), minval=-0.1, maxval=0.1)
        new_dim = tf.cast(tf.ceil((1 - zoom) * target_size[0]), dtype=tf.int32)

        image = tf.image.resize_image_with_crop_or_pad(image, new_dim, new_dim)

        image = tf.image.resize_images(
            image, target_size, method=tf.image.ResizeMethod.BILINEAR)

      if normalize:
        std = tf.constant(
            np.array([70.53946096, 51.71475228, 43.03428563]), dtype=tf.float32)
        std = tf.expand_dims(tf.expand_dims(std, axis=0), axis=0)

        mean = tf.constant(
            np.array([108.64628601, 75.86886597, 54.34005736]),
            dtype=tf.float32)
        mean = tf.expand_dims(tf.expand_dims(mean, axis=0), axis=0)

        image = (tf.cast(image, dtype=tf.float32) - mean) / std

      label = tf.reshape(label, [1])
      if input_type == ".jpeg":
        image = tf.reshape(image, target_size + [3])
      else:
        image = tf.reshape(image, target_size)

      return {'shape': shape, 'image': image}, label

    dataset = dataset \
        .apply(map_and_batch(_decode_and_augment_image, batch_size=batch_size, num_parallel_batches=nb_workers,
                             drop_remainder=True)) \
        .prefetch(nb_workers)

    # def _augment_images(example)

    return dataset
