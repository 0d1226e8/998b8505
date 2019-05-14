from abc import ABCMeta, abstractmethod


class _BaseOptIn:
  """Abstract class specifying the
  signatures of the optional APIs which
  may be framework dependent.
  """

  __metaclass__ = ABCMeta

  @abstractmethod
  def train_tf_input_fn(self):
    """Returns TensorFlow input function
    that iterates of the training set.

    Returns:
      Function () -> (features, labels)
        * `features`: TensorFlow tensor or dict
          of TensorFlow tensors representing
          the features of the dataset.
        * `labels`: TensorFlow tensor representing the labels.
    """

  @abstractmethod
  def validate_tf_input_fn(self):
    """Returns TensorFlow input function
    that iterates of the training set.

    Returns:
      Function () -> (features, labels)
        * `features`: TensorFlow tensor or dict of TensorFlow tensors representing
            the features of the dataset.
        * `labels`: TensorFlow tensor representing the labels.
    """
