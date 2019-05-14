import os
from model import model_fn
import tensorflow as tf
from bdlb.tasks import DiabetesMedium
import numpy as np

VAL_REPORT_FREQ = 1

tf.set_random_seed(1241)

# Load experiment object for Medium benchmark
exp = DiabetesMedium('./data/diabetes')

BATCH_SIZE = 16
batches_per_epoch = exp.train_ds.y.size // BATCH_SIZE
tf.logging.set_verbosity(tf.logging.INFO)

# Set model dir where to save checkpoints and report
model_dir = './data/bbp_medium'

# Create TF estimator
params = {
    'initial_conv_units': 16,
    'dense_units': 512,
    'nr_of_dense_layers': 0,
    'initial_lr': 0.01,
    'decay_rate': 0.98,
    'batches_per_epoch': batches_per_epoch,
    'reg_loss_factor': 20,
    'nr_samples': exp.train_ds.y.size
}
classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, params=params)

# Create estimator that takes images and produces prediction and uncertainties
external_batch_size = 500  # Number of images to give to estimator at once
def estimate(images):
  images = images.astype(np.float32)

  input_fun = tf.estimator.inputs.numpy_input_fn({'image': images},
                                                 None,
                                                 batch_size=BATCH_SIZE,
                                                 num_epochs=40,
                                                 shuffle=False,
                                                 num_threads=1)

  mc_preds = np.array(
      list(
          map(lambda x: x['probabilities'],
              classifier.predict(input_fn=input_fun))))
  mc_preds = mc_preds.reshape((-1, external_batch_size))

  return mc_preds.mean(axis=0), mc_preds.std(axis=0)

# Alternatingly train and evaluate estimator, change reg_loss_factor parameter when eval auc
# reaches 0.65
changed = False
for i in range(1, 150, 3):
  print("STARTING EPOCH", i, 'to', i + 2)
  print("=====================================")
  classifier.train(
      input_fn=exp.train_tf_input_fn(
          batch_size=BATCH_SIZE, nb_workers=20, nr_epochs=3))
  eval_result = classifier.evaluate(
      input_fn=exp.validate_tf_input_fn(batch_size=BATCH_SIZE, nb_workers=20))
  if eval_result['auc'] > 0.65 and not changed:
    params['reg_loss_factor'] = 1
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, params=params)
    changed = True

  # if i % VAL_REPORT_FREQ == 0:
  #   # Evaluate estimator using experiment object
  #   report = exp.generate_report(estimate,
  #                                mode="eval",
  #                                model_class='MFVI_BNN',
  #                                dataset_type='medium',
  #                                batch_size=external_batch_size)
  #
  #   # Generate report
  #   tex = report.to_latex(
  #         output_file="tmp/results/diabetes/medium/MFVI_tf/val_epoch{}_report.tex".
  #         format(i),
  #         title="Diabetes medium benchmark with MFVI baseline")
  #   print(tex)

# Evaluate estimator using experiment object
report = exp.generate_report(estimate,
                             mode="test",
                             model_class='MFVI_BNN',
                             dataset_type='medium',
                             batch_size=external_batch_size)

# Generate report
report_dir = os.path.join(model_dir, 'report')
report.to_latex(
    'MFVI Medium', output_file=os.path.join(report_dir, 'report_mfvi_med.tex'))
