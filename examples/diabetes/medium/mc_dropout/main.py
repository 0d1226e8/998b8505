import os

import numpy as np
import tensorflow as tf
from tqdm import trange

from bdlb.tasks import DiabetesMedium
from model import vggish_fn


def main(argv):

  #############
  # Benchmark #
  #############

  # Load experiment object for Medium benchmark
  dtask = DiabetesMedium('./data/diabetes')

  ####################
  # TensorFlow Setup #
  ####################

  tf.set_random_seed(1234)
  tf.logging.set_verbosity(tf.logging.ERROR)

  FLAGS = tf.flags.FLAGS

  tf.flags.DEFINE_string(
      "output_dir",
      None,
      "Location to write training files to",
  )

  # Some TF settings
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  estimator_config = tf.estimator.RunConfig(session_config=session_config)

  ##################
  # Hyperparmeters #
  ##################

  NUM_EVALUATIONS = 10  # Numbers to run evaluation with stochastic metrics
  VAL_REPORT_FREQ = 10

  BATCH_SIZE = 16
  EXTERNAL_BATCH_SIZE = 500  # Number of images to give to estimator at once
  params = {
      "dropout_conv": 0.1,
      "dropout_dense": 0.2,
      "initial_conv_units": 16,
      "dense_units": 512,
      "nr_of_dense_layers": 0,
      "lr": 0.001,
      "l2_reg": 5e-5
  }

  #########
  # Model #
  #########

  # Set model dir where to save checkpoints and report
  model_dir = FLAGS.output_dir

  # VGG-ish model with dropout
  classifier = tf.estimator.Estimator(
      model_fn=vggish_fn,
      model_dir=model_dir,
      config=estimator_config,
      params=params,
  )

  #########################
  # Uncertainty Estimator #
  #########################

  def estimate(images):
    """Uncertainty estimator function.

      Args:
        images: Input features, `NumPy` array.

      Returns:
        mean: Predictive mean.
        uncertainty: Uncertainty in prediction.
      """
    # Type casting.
    images = images.astype(np.float32)

    # TensorFlow Estimator API
    input_fun = tf.estimator.inputs.numpy_input_fn(
        {'image': images},
        None,
        batch_size=EXTERNAL_BATCH_SIZE,
        num_epochs=20,
        shuffle=False,
        num_threads=1,
    )

    # MC samples from approximate posterior
    mc_preds = np.array(
        list(
            map(lambda x: x["probabilities"],
                classifier.predict(input_fn=input_fun))))
    mc_preds = mc_preds.reshape((-1, min(len(images), EXTERNAL_BATCH_SIZE)))
    _MC_samples = np.concatenate(
        [1. - mc_preds[:, :, None], mc_preds[:, :, None]], -1)

    # Predictive mean
    mean_expected_p = mc_preds.mean(axis=0)

    # Predictive entropy
    expected_p = np.mean(_MC_samples, axis=0)
    entropy_expected_p = -np.sum(expected_p * np.log(expected_p + 1e-10),
                                 axis=-1)

    return mean_expected_p, entropy_expected_p

  #################
  # Training Loop #
  #################

  # Alternatingly train and evaluate estimator
  for i in trange(1, 300, 3):
    # Training operation.
    classifier.train(input_fn=dtask.train_tf_input_fn(
        batch_size=BATCH_SIZE,
        nb_workers=20,
        nr_epochs=3,
    ))
    # Evaluation on validation dataset.
    classifier.evaluate(input_fn=dtask.validate_tf_input_fn(
        batch_size=BATCH_SIZE,
        nb_workers=20,
    ))

    # if i % VAL_REPORT_FREQ == 0:
    #   # Evaluate estimator with respect to stochastic metrics.
    #   report = dtask.generate_report(estimate,
    #                                  mode="eval",
    #                                  model_class='MC_Dropout_BNN',
    #                                  dataset_type='medium',
    #                                  batch_size=EXTERNAL_BATCH_SIZE)
    #   # Generate report.
    #   tex = report.to_latex(
    #       output_file=
    #       "tmp/results/diabetes/medium/mc_dropout_tf/val_epoch{i}_report.tex".
    #       format(i=i),
    #       title="Diabetes medium benchmark with MC Dropout baseline")
    #   print(tex)

  ##############
  # Evaluation #
  ##############

  for n in range(NUM_EVALUATIONS):
    # Evaluate estimator on test dataset.
    report = dtask.generate_report(estimate,
                                   mode="test",
                                   model_class='MC_Dropout_BNN',
                                   dataset_type='medium',
                                   batch_size=EXTERNAL_BATCH_SIZE)

    # Generate report.
    tex = report.to_latex(
        output_file="tmp/results/diabetes/medium/mc_dropout_tf/report_{n}.tex".
        format(n=n),
        title="Diabetes medium benchmark with MC Dropout baseline")
    print(tex)


if __name__ == '__main__':
  tf.app.run(main)