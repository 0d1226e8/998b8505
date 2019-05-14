"""
Largely inspired by https://github.com/chleibig/disease-detection
"""

from __future__ import division

from collections import namedtuple
from pathos import multiprocessing
import numpy as np


def bootstrap(data, fun, n_resamples=10000, alpha=0.05):
  """
    Compute confidence interval for values of function fun on data_preprocess_scripts

    Args:
        data : list of numpy arrays
            Each numpy array will be subsampled and then passed to fun.

        fun :
            Function taking len(data_preprocess_scripts) numpy arrays -> float

        n_resamples : int, 10000 is default
            Number of times to resample data_preprocess_scripts to produce intervals.

        alpha : float, 0.5 is default
            Confidence level parameter for confidence intervals.

    """
  assert isinstance(data, list)
  n_samples = len(data[0])
  idx = np.random.randint(0, n_samples, (n_resamples, n_samples))

  def select(data, sample):
    return [d[sample] for d in data]

  def evaluate(sample):
    return fun(*select(data, sample))

  pool = multiprocessing.Pool(multiprocessing.cpu_count())
  values = pool.map(evaluate, idx)
  pool.terminate()

  idx = idx[np.argsort(values, axis=0, kind='mergesort')]
  values = np.sort(values, axis=0, kind='mergesort')

  stat = namedtuple('stat', ['value', 'index'])
  low = stat(
      value=values[int((alpha / 2.0) * n_resamples)],
      index=idx[int((alpha / 2.0) * n_resamples)])
  high = stat(
      value=values[int((1 - alpha / 2.0) * n_resamples)],
      index=idx[int((1 - alpha / 2.0) * n_resamples)])

  return low, high
