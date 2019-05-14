import functools

import tensorflow as tf
import tensorflow.keras.layers as tfkl


def vggish_fn(features, labels, mode, params):
  """Tensorflow Estimator API VGG-ish model function."""
  with tf.variable_scope(
      "model",
      initializer=tf.keras.initializers.he_normal(),
  ):
    # Fetch data.
    inputs = features["image"]
    # Hyperparameters.
    dropout_rate_conv = params["dropout_conv"]
    dropout_rate_dense = params["dropout_dense"]
    initial_conv_units = params["initial_conv_units"]
    dense_units = params["dense_units"]
    nr_of_dense_layers = params["nr_of_dense_layers"]
    lr = params["lr"]
    l2_reg = params["l2_reg"]
    # L2 Regularizer.
    l2regularization = tf.contrib.layers.l2_regularizer(l2_reg)

    if mode == tf.estimator.ModeKeys.EVAL:
      apply_dropout = False
    else:
      apply_dropout = True

    # Helper functions for layers.
    leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.2)

    def _conv_block(x,
                    conv2d_units,
                    conv2d_strides,
                    num_layers,
                    max_pooling=True):
      """Convolutional block, `num_layers` x (`Conv2D` + `Dropout`) [+ `MaxPooling2D`]."""
      for _ in range(num_layers):
        x = tfkl.Conv2D(
            filters=conv2d_units,
            kernel_size=3,
            strides=conv2d_strides,
            activation=leaky_relu,
            padding="same",
            kernel_regularizer=l2regularization,
        )(x)
        x = tfkl.Dropout(rate=dropout_rate_conv)(x, training=apply_dropout)
      if max_pooling:
        x = tfkl.MaxPooling2D(
            pool_size=3,
            strides=(2, 2),
            padding="same",
        )(x)
      return x

    # Convolutional layers.
    t = _conv_block(x=inputs,
                    conv2d_units=initial_conv_units,
                    conv2d_strides=(2, 2),
                    num_layers=1,
                    max_pooling=True)
    t = _conv_block(x=t,
                    conv2d_units=initial_conv_units,
                    conv2d_strides=(1, 1),
                    num_layers=2,
                    max_pooling=True)
    t = _conv_block(x=t,
                    conv2d_units=initial_conv_units * 2,
                    conv2d_strides=(1, 1),
                    num_layers=2,
                    max_pooling=True)
    t = _conv_block(x=t,
                    conv2d_units=initial_conv_units * 4,
                    conv2d_strides=(1, 1),
                    num_layers=4,
                    max_pooling=True)
    t = _conv_block(x=t,
                    conv2d_units=initial_conv_units * 8,
                    conv2d_strides=(1, 1),
                    num_layers=4,
                    max_pooling=False)

    # Global pooling layers.
    mean_pool = tfkl.GlobalAvgPool2D()(t)
    max_pool = tfkl.GlobalMaxPool2D()(t)
    t = tfkl.Concatenate([mean_pool, max_pool], axis=1)

    # Fully connected layers.
    for _ in range(nr_of_dense_layers):
      t = tfkl.Dropout(rate=dropout_rate_dense)(t, training=apply_dropout)
      t = tfkl.Dense(
          dense_units,
          activation=leaky_relu,
          kernel_regularizer=l2regularization,
      )(t)

    # Output layer.
    logits = tf.layers.Dense(1, kernel_regularizer=l2regularization)(t)

    # Forward pass.
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

    # Loss function.
    loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, dtype=tf.float32),
        logits=logits,
    )
    loss_ce = tf.reduce_mean(loss_vec)
    loss_reg = tf.losses.get_regularization_loss()
    loss = loss_ce + loss_reg
    pred_sig = tf.nn.sigmoid(logits)

    # Metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=tf.to_int64(pred_sig > 0.5))
    accuracy_per_class = tf.metrics.mean_per_class_accuracy(
        labels=labels, predictions=tf.to_int64(pred_sig > 0.5), num_classes=2)
    loss_ce_mean = tf.metrics.mean(loss_ce)
    reg_loss_mean = tf.metrics.mean(loss_reg)
    auc = tf.metrics.auc(labels=labels, predictions=pred_sig)

    # Training operation.
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)
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
          training_hooks=[logging_hook],
      )

    # Evaluation operation.
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
          },
      )
