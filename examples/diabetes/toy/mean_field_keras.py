import matplotlib
matplotlib.use('agg')

import numpy as np

from bdlb.tasks import DiabetesToy
import tensorflow as tf
import tensorflow_probability as tfp

# Hyperparameters
HPARAMS = {
    "batch_size": 100,
    "num_epochs": 25,
    "l2_reg": tf.keras.regularizers.l2(1e-3),
    "num_MC_samples": 20
}

VAL_REPORT_FREQ = 1

# Load experiment object for Toy benchmark
dtask = DiabetesToy("data/diabetes/")

# Define model
inputs = tf.keras.Input(shape=(512,))
x = tfp.layers.DenseReparameterization(1024, activation="relu")(inputs)
x = tfp.layers.DenseReparameterization(1024, activation="relu")(x)
x = tfp.layers.DenseReparameterization(1024, activation="relu")(x)
outputs = tfp.layers.DenseReparameterization(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="adam")
model.summary()

def predict(x):
  """Uncertainty estimator function.

  Args:
    x: Input features, `NumPy` array.

  Returns:
    mean: Predictive mean.
    uncertainty: Uncertainty in prediction.
  """
  MC_samples = [model.predict(x) for _ in range(HPARAMS["num_MC_samples"])]
  MC_samples = np.array(MC_samples)
  _MC_samples = np.concatenate([1. - MC_samples, MC_samples], -1)
  expected_p = np.mean(_MC_samples, axis=0)
  entropy_expected_p = -np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
  return np.mean(MC_samples, 0), entropy_expected_p

# Training & evaluation loops
for epoch in range(HPARAMS["num_epochs"]):
  with dtask.iterate_train(batch_size=HPARAMS["batch_size"]) as it:
    for x, y in it:
      model.fit(x, y, batch_size=1, verbose=0)

  label_list = []
  predictions = []
  with dtask.iterate_validation(batch_size=HPARAMS["batch_size"]) as it:
    for x, y in it:
      label_list.append(y)
      pred_act = model.predict(x)
      predictions.append(pred_act)

  print(
      dtask.metrics_deterministic(
          np.concatenate(label_list), np.concatenate(predictions)))

# Report generation
report = dtask.generate_report(predict,
                               mode="test",
                               model_class='MFVI_BNN',
                               dataset_type='toy',
                               batch_size=HPARAMS["batch_size"])
tex = report.to_latex(
    output_file="tmp/results/diabetes/toy/mean_field_keras/report.tex",
    title="Diabetes toy benchmark with Mean Field Variational Inference baseline"
)
print(tex)
