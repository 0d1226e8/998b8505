from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from ...common.report.bootstrap import bootstrap
from ...common.report.roc_curve_plot import roc_curve_plot

plt.ion()
# sns.set_context('paper', font_scale=1.5)
# sns.set_style('whitegrid')

FIGURE_WIDTH = 8.27  # 8.27 inch corresponds to A4


def argmax_labels(probs):
  """
  Helper function for calculating accuracy.

  Args:
      probs : numpy array of floats
          Predictions.

  Returns:
      (probs >= 0.5).astype(int)
  """
  return (probs >= 0.5).astype(int)


def accuracy(y_true, probs):
  """
  Calculates accuracy when using probs to label y_true. Treshhold
  for positive label is taken to be 0.5.

  Args:
      y_true : numpy array of (0, 1)
          True labels.

      probs : numpy array of floats
          Predictions.

  Returns:
      Accuracy : float.
  """
  probs = probs.ravel()  # flattens array to 1D
  y_pred = argmax_labels(probs)
  assert len(y_true) == len(y_pred)
  return (y_true == y_pred).sum() / float(len(y_true))


def performance_over_uncertainty_tol(uncertainties,
                                     y,
                                     probs,
                                     measure,
                                     min_percentile,
                                     ci=False,
                                     n_bootstrap=10000):
  """
  Calculates performance measure when retaining a fraction of data_preprocess_scripts
  according to uncertainties.

  Args:
      uncertainties : numpy array
          Uncertainty of estimator in prediction of probs

      y : numpy array
          True labels.

      probs : numpy array
          Estimated probability label is true by estimator.

      measure : function (numpy array, numpy array) -> float
          Performance measure to calculate.

      min_percentile : float
          Minimum percentile to retain

      ci : boolean, False by default
          Whether to calculate confidence intervals. Warning, this may take a
          long time!

      n_bootstrap : int, 10000 by default
          Number of bootstrap iterations for calculating confidence intervals.

  Returns:
      3-tuple of:
          uncertainty_tol : numpy array of floats
              linspace of 100 floats between the min_percentile order
              statistic of uncertainies and the max of uncertainties

          frac_retain : numpy array of floats
              i-th element is the fraction of data_preprocess_scripts we retain if we discard
              elements s.t. uncertainties[i] > uncertainty_tol[i]

          p : numpy array of length 100 with custom dtype
              i-th element has values for keys:
                  'value': Value of performance measure if we discard elements
                  s.t. uncertainties > uncertainty_tol[i].

                  'low': Lower confidence bound of performance measure if we
                  discard elements s.t. uncertainties > uncertainty_tol[i].
                  Only present if ci is set to True.

                  'high': Higher confidence bound of performance measure if we
                  discard elements s.t. uncertainties > uncertainty_tol[i].
                  Only present if ci is set to True.
              Access element for example by p['value'][30]
  """
  uncertainty_tol, frac_retain, accept_idx = \
      sample_rejection(uncertainties, min_percentile)

  if ci:
    p = np.zeros((len(uncertainty_tol),),
                 dtype=[('value', 'float64'), ('low', 'float64'),
                        ('high', 'float64')])
  else:
    p = np.zeros((len(uncertainty_tol),), dtype=[('value', 'float64')])

  for i, ut in enumerate(uncertainty_tol):
    accept = accept_idx[i]
    p['value'][i] = measure(y[accept], probs[accept])
    if ci:
      low, high = bootstrap([y[accept], probs[accept]],
                            measure,
                            n_resamples=n_bootstrap,
                            alpha=0.05)
      p['low'][i] = low.value
      p['high'][i] = high.value

  return uncertainty_tol, frac_retain, p


def sample_rejection(uncertainties, min_percentile):
  """
  Calculates which uncertainties to retain when retaining a
  certain fraction of uncertainties.

  Args:
      uncertainties : numpy array
          Array of uncertainty measures
      min_percentile :
          Minimum percentile to retain

  Returns:
      3-tuple of:
          uncertainty_tol: numpy array of floats
              linspace of 100 floats between the min_percentile order
              statistic of uncertainies and the max of uncertainties

          frac_retain: numpy array of floats
              i-th element is the fraction of data_preprocess_scripts we retain if we discard
              elements s.t. uncertainties[i] > uncertainty_tol[i]

          accept_indices: list of numpy arrays of booleans
              i-th element is the mask of uncertainties that corresponds
              to uncertainties <= uncertainty_tol[i]
  """
  maximum = uncertainties.max()
  uncertainty_tol = np.linspace(
      np.percentile(uncertainties, min_percentile), maximum, 100)
  frac_retain = np.zeros_like(uncertainty_tol)
  n_samples = len(uncertainties)
  accept_indices = []
  for i, ut in enumerate(uncertainty_tol):
    accept = (uncertainties <= ut)
    accept_indices.append(accept)
    frac_retain[i] = accept.sum() / float(n_samples)

  return uncertainty_tol, frac_retain, accept_indices


def roc_intervals_plot(y,
                       y_score,
                       uncertainties,
                       name,
                       ci=False,
                       n_bootstrap=10000,
                       axis=None):
  """
  Generates plot of roc curves where 100%, 90%, 80% and 70% of
  data_preprocess_scripts is retained (where the indices with highest uncertainty is removed).

  Args:
      y : numpy array
          True labels.

      y_score : numpy array
          Estimated probability label is true by estimator.

      uncertainties : numpy array
          Uncertainty of estimator in prediction of y_score.

      name : str
          Name of estimator (used for labeling plot).

      ci : boolean, False by default
          Whether to calculate confidence intervals. Warning, this may take a
          long time!

      n_bootstrap : int, 10000 by default
          Number of bootstrap iterations for calculating confidence intervals.
  Returns:
      Matplotlib figure
  """
  fig = None
  if axis is None:
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    axis = fig.add_subplot(111)

  colors = sns.color_palette()

  axis.set_title('ROC curve under varying retained data rates')

  v_tol, frac_retain, _ = sample_rejection(uncertainties, 50)
  fractions = [0.9, 0.8, 0.7]
  for j, f in enumerate(fractions):
    thr = v_tol[frac_retain >= f][0]
    roc_curve_plot(
        y[uncertainties <= thr],
        y_score[uncertainties <= thr],
        color=colors[j + 4],
        legend='%d%% data retained' % (f * 100),
        ci=ci,
        n_bootstrap=n_bootstrap,
        axis=axis)

  roc_curve_plot(
      y,
      y_score,
      color=colors[-1],
      legend='no referral',
      n_bootstrap=n_bootstrap,
      axis=axis)
  if fig is not None:
    return fig
  return axis


def intervals_df(y,
                 y_score,
                 uncertainties,
                 name):
  """
  Generates pandas data_preprocess_scripts frame of auc and accuracy values where 100%, 90%, 80% and 70% of
  data is retained (where the indices with highest uncertainty is removed).

  Args:
      y : numpy array
          True labels.

      y_score : numpy array
          Estimated probability label is true by estimator.

      uncertainties : numpy array
          Uncertainty of estimator in prediction of y_score.

      name : str
          Name of estimator (used for labeling plot).

  Returns:
      Pandas data_preprocess_scripts frame
  """
  assert (isinstance(y, np.ndarray) and isinstance(y_score, list)
          and isinstance(uncertainties, list) and isinstance(name, list)) or \
         (isinstance(y, np.ndarray) and isinstance(y_score, np.ndarray) and
          isinstance(uncertainties, np.ndarray) and isinstance(name, str))

  if not isinstance(name, list):
    y_score = [y_score]
    uncertainties = [uncertainties]
    name = [name]

  assert len(y_score) == len(uncertainties) == len(name)

  fractions = [1.0, 0.9, 0.8, 0.7]
  data = {
      'fraction': fractions,
  }

  for i in range(len(name)):
    v_tol, frac_retain, _ = sample_rejection(uncertainties[i], 70)
    data[name[i] + ' auc'] = []
    data[name[i] + ' acc'] = []
    for j, f in enumerate(fractions):
      thr = v_tol[frac_retain >= f][0]
      data[name[i] + ' auc'].append(
          roc_auc_score(y[uncertainties[i] <= thr],
                        y_score[i][uncertainties[i] <= thr]))
      data[name[i] + ' acc'].append(
          accuracy(y[uncertainties[i] <= thr],
                   y_score[i][uncertainties[i] <= thr]))

  df = pd.DataFrame(data=data)
  df.set_index('fraction', inplace=True)
  return df


def measure_under_varying_referral_rate(data,
                                        names,
                                        measure,
                                        measure_name,
                                        ci=False,
                                        n_bootstrap=10000,
                                        axis=None):
  """
  Draws plot where the performance of y_scores compared to the true label ys is calculated
  for varying fractions of data_preprocess_scripts kept by ignoring the corresponding percentile of indices with
  highest uncertainty. May pass single numpy arrays or lists of numpy arrays (to compare different
  estimators).
  Args:
      ys : np array of ints or list of n numpy arrays of ints
          True labels.

      y_scores : np array of floats or list of n numpy arrays of floats
          Predicted label scores for each estimator.

      uncertainties : np array of floats or list of n numpy arrays of floats
          Uncertainties of each estimator.

      names : str or list of n strings
          Name of each estimator.

      measure : function (numpy array, numpy array) -> float
          Measure comparing the true labels ys with the y_scores.

      measure_name : str
          Name of measure to be used in title

      ci : boolean, False by default
          Whether to calculate confidence intervals. Warning, this may take a
          long time!

      n_bootstrap : int, 10000 by default
          Number of bootstrap iterations for calculating confidence intervals.

  Returns:
      matplotlib figure
  """
  ys = data['labels_binary_{}'.format(data['new_model'])]
  y_scores = data['predictions_{}'.format(data['new_model'])]
  uncertainties = data['uncertainties_{}'.format(data['new_model'])]

  # Ensure either y_scores, uncertainties and names are all list of equal length
  # or just just single numpy arrays. If not list, convert all to list with single
  # element.
  assert (isinstance(ys, np.ndarray) and isinstance(y_scores, list)
          and isinstance(uncertainties, list) and isinstance(names, list)) or \
         (isinstance(ys, np.ndarray) and isinstance(y_scores, np.ndarray) and
          isinstance(uncertainties, np.ndarray) and isinstance(names, str))
  if not isinstance(names, list):
    y_scores = [y_scores]
    uncertainties = [uncertainties]
    names = [names]
  assert len(y_scores) == len(uncertainties) == len(names)

  fig = None
  if axis is None:
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    axis = fig.add_subplot(111)

  colors = sns.color_palette()

  for i in range(len(names)):
    j = 1
    n = 0
    for m in data['model_classes']:
        acc_list = []
        acc_value_cum = 0
        while True:
            try:
                ys = data['labels_binary_{}_{}'.format(m, j)]
                y_scores = [data['predictions_{}_{}'.format(m, j)]]
                uncertainties = [data['uncertainties_{}_{}'.format(m, j)]]

                v_tol, frac_retain, acc = \
                    performance_over_uncertainty_tol(uncertainties[i], ys, y_scores[i],
                                                     measure, 50, ci=ci,
                                                     n_bootstrap=n_bootstrap)
                acc_list.append(acc)
                acc_value_cum += acc['value']
                j += 1
            except:
                if j > 2:
                    acc_value_mean = acc_value_cum / (j-1)
                    acc_value_std = np.std(np.array(acc_list, float), axis=0)
                else:
                    acc_value_mean = acc_value_cum / j
                    acc_value_std = 0
                j = 1
                break

        try:
            axis.plot(
                frac_retain, acc_value_mean, label='{}-{}'.format(m, names[i]), color=colors[i+n], linewidth=2)

            axis.fill_between(
                frac_retain, acc_value_mean, acc_value_mean-acc_value_std, color=colors[i+n], alpha=0.3)
            axis.fill_between(
                frac_retain, acc_value_mean+acc_value_std, acc_value_mean, color=colors[i+n], alpha=0.3)

            if ci:
              axis.fill_between(
                  frac_retain, acc['value'], acc['low'], color=colors[i+n], alpha=0.3)
              axis.fill_between(
                  frac_retain, acc['high'], acc['value'], color=colors[i+n], alpha=0.3)
            n += 1
        except:
            print('Not plotted for {}'.format(m))

  axis.set_title(measure_name + ' under varying retained data rates')

  axis.set_xlim(0.5, 1.0)
  axis.set_xlabel('Fraction of data retained')
  axis.set_ylabel(measure_name)
  axis.legend(loc='best')

  if fig is not None:
    return fig
  return axis


def level_plot(y_level,
               uncertainty,
               ax=None):
  fig = None
  if ax is None:
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    ax = fig.add_subplot(111)

  tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0)
  levels = OrderedDict([(0, 'no DR'), (1, 'mild DR'), (2, 'moderate DR'),
                        (3, 'severe DR'), (4, 'proliferative DR')])

  def rel_freq(y, k):
    return (y == k).sum() / float(len(y)) if len(y) > 0 else 0

  p = {
      level:
      np.array([rel_freq(y_level[~accept], level) for accept in accept_idx])
      for level in levels
  }
  cum = np.zeros_like(tol)

  # with sns.axes_style('white'):
  ax.set_title('proportion disease levels in referred datasets')

  colors = {level: sns.color_palette("Blues")[level] for level in levels}
  for level in levels:
    ax.fill_between(
        tol,
        p[level] + cum,
        cum,
        color=colors[level],
        label='%d: %s' % (level, levels[level]))
    if level == 1:
      ax.plot(tol, p[level] + cum, color='k', label='healthy/diseased boundary')
    cum += p[level]

  ax.set_xlim(min(tol), max(tol))
  ax.set_ylim(0, 1)

  ax.set_xlabel('tolerated model uncertainty')
  ax.set_ylabel('relative proportions within referred dataset')
  ax.legend(loc='lower left')
  ax.grid(None)

  if fig is not None:
    return fig
  return ax


class EvaluationResult:

  def __init__(self, acc, auc):
    self.acc = acc
    self.auc = auc

  def __repr__(self):
    return "<EvaluationResult acc:{}, auc:{}>".format(self.acc, self.auc)

  def __str__(self):
    return "<Accuracy: {}, auc:{}>".format(self.acc, self.auc)
