try:
  from keras.preprocessing.image import ImageDataGenerator
except ImportError:
  from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import numpy as np

AUGMENTATION_PARAMS = {
    'featurewise_center': False,
    'samplewise_center': False,
    'featurewise_std_normalization': False,
    'samplewise_std_normalization': False,
    'zca_whitening': False,
    'rotation_range': 180.,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.,
    'zoom_range': 0.10,
    'channel_shift_range': 0.,
    'fill_mode': 'constant',
    'cval': 0.,
    'horizontal_flip': True,
    'vertical_flip': True
}


def standard_normalize(image):
  """
    Normalize image to have zero mean and unit variance.
    Subtracts channel MEAN and divides by channel STD

    Args:
        image : numpy array
            Array representing image

    Returns:
        image : numpy array
            Array representing image
    """

  # channel standard deviations (calculated by team o_O during Kaggle competition)
  STD = np.array([70.53946096, 51.71475228, 43.03428563])
  # channel means (calculated by team o_O during Kaggle competition)
  MEAN = np.array([108.64628601, 75.86886597, 54.34005737])

  return np.divide(
      np.subtract(image, MEAN[np.newaxis, np.newaxis, :]),
      STD[np.newaxis, np.newaxis, :])


def image_preprocess(data_format="channels_last", augment=True):
  if augment:
    AUGMENTATION_PARAMS['data_format'] = data_format
    image_data_generator = ImageDataGenerator(**AUGMENTATION_PARAMS)

  def im_preprocess(img):
    img = standard_normalize(img)
    if augment:
      img = image_data_generator.random_transform(img)
    return img

  return im_preprocess
