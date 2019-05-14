# Copyright 2018 BDL Benchmarks Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib
matplotlib.use('agg')

import os
from functools import partial

import numpy as np
import tensorflow as tf

from bdlb.tasks import DiabetesRealWorld

VAL_REPORT_FREQ = 1


def model_fn(features, labels, mode, params):
  inputs = features["image"]

  dropout_rate_conv = params["dropout_conv"]
  dropout_rate_dense = params["dropout_dense"]
  initial_conv_units = params["initial_conv_units"]
  dense_units = params["dense_units"]
  nr_of_dense_layers = params["nr_of_dense_layers"]
  initial_lr = params["initial_lr"]
  decay_rate = params["decay_rate"]
  batches_per_epoch = params["batches_per_epoch"]
  l2_reg = params["l2_reg"]

  l2regularization = tf.contrib.layers.l2_regularizer(l2_reg)

  if mode == tf.estimator.ModeKeys.EVAL:
    apply_dropout = False
  else:
    apply_dropout = True

  leaky_relu = partial(tf.nn.leaky_relu, alpha=0.2)

  learning_rate = tf.train.exponential_decay(
      initial_lr,  # Base learning rate.
      tf.train.get_or_create_global_step(),  # Current index into the dataset.
      batches_per_epoch,  # Decay step.
      decay_rate,  # Decay rate.
      staircase=True)

  t = tf.layers.Conv2D(
      initial_conv_units,
      3,
      strides=(2, 2),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(inputs)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(t)

  t = tf.layers.Conv2D(
      initial_conv_units,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.Conv2D(
      initial_conv_units,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(t)

  t = tf.layers.Conv2D(
      initial_conv_units * 2,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.Conv2D(
      initial_conv_units * 2,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(t)

  t = tf.layers.Conv2D(
      initial_conv_units * 4,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)

  t = tf.layers.Conv2D(
      initial_conv_units * 4,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)

  t = tf.layers.Conv2D(
      initial_conv_units * 4,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)

  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.Conv2D(
      initial_conv_units * 4,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)
  t = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(t)

  t = tf.layers.Conv2D(
      initial_conv_units * 8,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)

  t = tf.layers.Conv2D(
      initial_conv_units * 8,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)

  t = tf.layers.Conv2D(
      initial_conv_units * 8,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)
  t = tf.layers.dropout(t, rate=dropout_rate_conv, training=apply_dropout)

  t = tf.layers.Conv2D(
      initial_conv_units * 8,
      3,
      strides=(1, 1),
      activation=leaky_relu,
      padding="same",
      kernel_regularizer=l2regularization)(t)

  mean_pool = tf.keras.layers.GlobalAvgPool2D()(t)
  max_pool = tf.keras.layers.GlobalMaxPool2D()(t)

  t = tf.concat([mean_pool, max_pool], axis=1)

  for _ in range(nr_of_dense_layers):
    t = tf.layers.dropout(t, rate=dropout_rate_dense, training=apply_dropout)
    t = tf.layers.Dense(
        dense_units, activation=leaky_relu,
        kernel_regularizer=l2regularization)(t)

  logits = tf.layers.Dense(1, kernel_regularizer=l2regularization)(t)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "label": tf.to_int64(tf.nn.sigmoid(logits) > 0.5),
        "probabilities": tf.nn.sigmoid(logits)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })

  loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.cast(labels, dtype=tf.float32), logits=logits)
  loss_ce = tf.reduce_mean(loss_vec)
  loss_reg = tf.losses.get_regularization_loss()
  loss = loss_ce + loss_reg
  pred_sig = tf.nn.sigmoid(logits)

  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.to_int64(pred_sig > 0.5))
  accuracy_per_class = tf.metrics.mean_per_class_accuracy(
      labels=labels, predictions=tf.to_int64(pred_sig > 0.5), num_classes=2)
  loss_ce_mean = tf.metrics.mean(loss_ce)
  reg_loss_mean = tf.metrics.mean(loss_reg)
  auc = tf.metrics.auc(labels=labels, predictions=pred_sig)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, 0.9, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimize_op = optimizer.minimize(loss,
                                       tf.train.get_or_create_global_step())
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])

    logging_hook = tf.train.LoggingTensorHook(
        {
            "regularization_loss": reg_loss_mean[1],
            "cross_entropy_loss": loss_ce_mean[1],
            "accuracy": accuracy[1],
            "auc": auc[1],
            "per_class_accuracy": accuracy_per_class[1]
        },
        every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimize_op,
        training_hooks=[logging_hook])

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={
            "accuracy": accuracy,
            "accuracy_per_class": accuracy_per_class,
            "auc": auc,
            "loss_cross_entropy": loss_ce_mean,
            "regularization_loss": reg_loss_mean,
        })


# Load experiment object for RealWorld benchmark
dtask = DiabetesRealWorld()

BATCH_SIZE = 64
batches_per_epoch = dtask.train_ds.y.size // BATCH_SIZE
tf.logging.set_verbosity(tf.logging.INFO)

# Set model dir where to save checkpoints and report
model_dir = "./tmp/models/diabetes/realworld/mc_dropout_tf"

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
estimator_config = tf.estimator.RunConfig(session_config=session_config)

# Create TF estimator
classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    config=estimator_config,
    params={
        "dropout_conv": 0.1,
        "dropout_dense": 0.2,
        "initial_conv_units": 32,
        "dense_units": 512,
        "nr_of_dense_layers": 0,
        "initial_lr": 0.05,
        "decay_rate": 0.98,
        "batches_per_epoch": batches_per_epoch,
        "l2_reg": 5e-5
    })

# Create estimator that takes images and produces prediction and uncertainties
external_batch_size = 100  # Number of images to give to estimator at once
def predict(images):
  """Uncertainty estimator function.

  Args:
    x: Input features, `NumPy` array.

  Returns:
    mean: Predictive mean.
    uncertainty: Uncertainty in prediction.
  """
  images = images.astype(np.float32)

  input_fun = tf.estimator.inputs.numpy_input_fn({"image": images},
                                                 None,
                                                 batch_size=external_batch_size,
                                                 num_epochs=40,
                                                 shuffle=False,
                                                 num_threads=1)

  mc_preds = np.array(
      list(
          map(lambda x: x["probabilities"],
              classifier.predict(input_fn=input_fun))))
  mc_preds = mc_preds.reshape((-1, min(len(images), external_batch_size)))

  _MC_samples = np.concatenate(
      [1. - mc_preds[:, :, None], mc_preds[:, :, None]], -1)
  expected_p = np.mean(_MC_samples, axis=0)
  entropy_expected_p = -np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)

  return mc_preds.mean(axis=0), entropy_expected_p

# Alternatingly train and evaluate estimator
for i in range(1, 151, 3):
  tf.logging.info("STARTING EPOCH " + str(i) + " to " + str(i + 2))
  tf.logging.info("=====================================")
  classifier.train(
      input_fn=dtask.train_tf_input_fn(
          batch_size=BATCH_SIZE, nb_workers=20, nr_epochs=3))
  classifier.evaluate(
      input_fn=dtask.validate_tf_input_fn(batch_size=BATCH_SIZE, nb_workers=20))

  # if i % VAL_REPORT_FREQ == 0:
  #   # Evaluate estimator on stochastic metrics
  #   report = dtask.generate_report(predict,
  #                                  mode="eval",
  #                                  model_class='MC_Dropout_BNN',
  #                                  dataset_type='realworld',
  #                                  batch_size=external_batch_size)
  #   # Generate report
  #   tex = report.to_latex(
  #       output_file="tmp/results/diabetes/realworld/mc_dropout_tf/val_epoch{}_report.tex".
  #       format(i),
  #       title="Diabetes realworld benchmark with MC Dropout baseline")
  #   print(tex)

# Evaluate estimator using experiment object
report = dtask.generate_report(predict,
                               mode="test",
                               model_class='MC_Dropout_BNN',
                               dataset_type='realworld',
                               batch_size=external_batch_size)

# Generate report
tex = report.to_latex(
    output_file="tmp/results/diabetes/realworld/mc_dropout_tf/report.tex",
    title="Diabetes real-world benchmark with MC Dropout baseline")
print(tex)
