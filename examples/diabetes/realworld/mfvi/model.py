from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp


def prior_fn(dtype, shape, name, trainable, add_variable_fn):
  del name, trainable, add_variable_fn  # unused
  tfd = tfp.distributions
  dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(1))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def model_fn(features, labels, mode, params):
  inputs = features['image']

  initial_conv_units = params['initial_conv_units']
  dense_units = params['dense_units']
  nr_of_dense_layers = params['nr_of_dense_layers']
  initial_lr = params['initial_lr']
  decay_rate = params['decay_rate']
  batches_per_epoch = params['batches_per_epoch']
  nr_samples = params['nr_samples']
  reg_loss_factor = params['reg_loss_factor']

  kernel_initializer = tfp.layers.default_mean_field_normal_fn(
      loc_initializer=tf.initializers.variance_scaling(
          scale=0.95, mode="fan_in", distribution="uniform"),
      untransformed_scale_initializer=tf.random_normal_initializer(
          mean=-4., stddev=0.5))

  leaky_relu = partial(tf.nn.leaky_relu, alpha=0.2)

  learning_rate = tf.train.exponential_decay(
      initial_lr,  # Base learning rate.
      tf.train.get_or_create_global_step(),  # Current index into the dataset.
      batches_per_epoch,  # Decay step.
      decay_rate,  # Decay rate.
      staircase=True)

  kl_div = tfp.distributions.kl_divergence

  # Must use Keras models to capture KL divergence terms from TF probability layers
  model1 = tf.keras.Sequential([
      tfp.layers.Convolution2DFlipout(
          initial_conv_units,
          3,
          strides=(2, 2),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same'),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same'),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 2,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 2,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same'),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 4,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 4,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 4,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 4,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same'),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 8,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 8,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 8,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
      tfp.layers.Convolution2DFlipout(
          initial_conv_units * 8,
          3,
          strides=(1, 1),
          activation=leaky_relu,
          padding='same',
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)),
  ])

  intermediary_output = model1(inputs)

  mean_pool = tf.keras.layers.GlobalAvgPool2D()(intermediary_output)
  max_pool = tf.keras.layers.GlobalMaxPool2D()(intermediary_output)

  intermediary_input = tf.concat([mean_pool, max_pool], axis=1)

  output_layers = []
  for _ in range(nr_of_dense_layers):
    output_layers.append(
        tfp.layers.DenseFlipout(
            dense_units,
            activation=leaky_relu,
            kernel_posterior_fn=kernel_initializer,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)))

  output_layers.append(
      tfp.layers.DenseFlipout(
          1,
          kernel_posterior_fn=kernel_initializer,
          kernel_prior_fn=prior_fn,
          kernel_divergence_fn=lambda q, p, ignore: kl_div(q, p)))

  model2 = tf.keras.Sequential(output_layers)

  logits = model2(intermediary_input)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'label': tf.to_int64(tf.nn.sigmoid(logits) > 0.5),
        'probabilities': tf.nn.sigmoid(logits)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  global_step = tf.train.get_or_create_global_step()

  loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.cast(labels, dtype=tf.float32), logits=logits)
  loss_ce = tf.reduce_mean(loss_vec)
  loss_reg = (
      sum(model1.losses) + sum(model2.losses)) / nr_samples / reg_loss_factor
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
      with tf.control_dependencies([tf.assign(global_step, global_step + 1)]):
        optimize_op = optimizer.minimize(loss)

    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_auc', auc[1])

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
            'accuracy': accuracy,
            'accuracy_per_class': accuracy_per_class,
            'auc': auc,
            "loss_cross_entropy": loss_ce_mean,
            "regularization_loss": reg_loss_mean,
        })
