"""
Largely inspired by https://github.com/chleibig/disease-detection
"""

from __future__ import division

import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score

from .bootstrap import bootstrap


def roc_curve_plot(y_true,
                   y_score,
                   legend,
                   pos_label=1,
                   ci=False,
                   n_bootstrap=10000,
                   color=None,
                   axis=None):
  """Compute and plot receiver operating characteristic (ROC)

    Args:
        y_true : array of int, shape = [n_samples]
            True binary labels

        y_score : array, shape = [n_samples]
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decision.

        legend : string
            plot legend

        pos_label : int
            Label considered as positive and others are considered negative.

        ci : boolean, False by default
            Plot confidence intervals by bootstrapping. This causes the ROC
            curve to take much longer to compute

        n_bootstrap : int, 10000 by default
            If confidence intervals are drawn, the number of simulations to run.

        color:
            matplotlib color arg or rgba tuple
    """
  assert y_score.ndim == 1, 'y_score should be of shape (n_samples,)'
  assert len(y_true) == len(y_score), \
      'y_true and y_score must both be n_samples long'

  fdr, tdr, _ = roc_curve(y_true, y_score, pos_label=pos_label)

  plot_target = axis if axis else plt

  plot_target.plot(fdr, tdr, color=color, label=legend, linewidth=2)

  if ci:
    low, high = bootstrap([y_true, y_score],
                          roc_auc_score,
                          n_resamples=n_bootstrap,
                          alpha=0.05)

    fdr_low, tdr_low, _ = roc_curve(
        y_true[low.index], y_score[low.index], pos_label=pos_label)
    fdr_high, tdr_high, _ = roc_curve(
        y_true[high.index], y_score[high.index], pos_label=pos_label)
    interpolate_low = interpolate.interp1d(fdr_low, tdr_low, kind='nearest')
    interpolate_high = interpolate.interp1d(fdr_high, tdr_high, kind='nearest')

    plot_target.fill_between(
        fdr, interpolate_high(fdr), tdr, color=color, alpha=0.3)
    plot_target.fill_between(
        fdr, tdr, interpolate_low(fdr), color=color, alpha=0.3)

  plot_target.plot([0, 1], [0, 1], 'k--')
  if axis is None:
    plot_target.xlim([0.0, 1.0])
    plot_target.ylim([0.0, 1.0])
    plot_target.xlabel('1 - specificity')
    plot_target.ylabel('sensitivity')
    plot_target.legend(loc="lower right")
  else:
    plot_target.set_xlim([0.0, 1.0])
    plot_target.set_ylim([0.0, 1.0])
    plot_target.set_xlabel('1 - specificity')
    plot_target.set_ylabel('sensitivity')
    plot_target.legend(loc="lower right")
