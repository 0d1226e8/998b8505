from abc import ABCMeta, abstractmethod


class _BaseBenchmarkCore:
  """Abstract class specifying the signatures
  of the core APIs which are framework agnostic.
  """

  __metaclass__ = ABCMeta

  def __init__(self):
    self.nr_train_samples = None
    self.nr_test_samples = None
    self.nr_eval_samples = None

  @abstractmethod
  def iterate_train(self):
    """Returns iterator over training set, that
    iterates over a single epoch.

    Returns:
      Python iterator, recommend using
      bdlb.common.data_loading.dataset
      for implementation
    """

  @abstractmethod
  def iterate_validation(self):
    """Returns iterator over validation set, that
    iterates over a single epoch.

    Returns:
      Python iterator, recommend using
      bdlb.common.data_loading.dataset
      for implementation
    """

  @staticmethod
  @abstractmethod
  def metrics_deterministic(labels, predictions):
    """Static helper function that takes a NumPy
    array of labels and predictions and returns
    an object containing metrics.

    Args:
      labels: True labels.
      predictions: Predicted values.

    Returns:
      Printable object with relevant statistics.
    """

  @abstractmethod
  def generate_report(self, estimator):
    """Function that evaluates the performance of
    an estimator over a dataset.

    Args:
      estimator: Function that takes a NumPy
        array batch of inputs (e.g. images) and returns a
        batch of predictions and uncertainties.

    Returns:
      Report object, containing relevant metrics
      and figures, and has a function to_latex
      which a latex project generating a report.
    """
