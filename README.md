# Bayesian Deep Learning Benchmarks

**Bayesian Deep Learning Benchmarks** (BDL Benchmarks or `bdlb` for short),
is an open-source framework that aims to bridge the gap between the design
of deep probabilistic machine learning models and their application to
real-world problems. BDL Benchmarks:

* provides a transparent, modular and consistent interface
  for the evaluation of deep probabilistic models on a variety of _downstream tasks_;
* abstracts away the expert-knowledge and eliminates the boilerplate steps necessary
  for running experiments on real-world datasets;
* makes it easy to compare the performance of new models against _baselines_,
  models that have been well-adopted by the machine learning community, under a
  fair setting (e.g., computational resources, model sizes, datasets).
* provides reference implementations of baseline models
  (e.g., Monte Carlo Dropout Inference [[1]](docs/Citations.md#Yarin-2015),
  Mean Field Variational Inference [[2]](docs/Citations.md#Peterson-1987),
  Model Ensembling) enabling rapid prototyping and easy development.
* integrates with the SciPy ecosystem (i.e., `NumPy`, `Pandas`, `Matplotlib`) and hence is
  independent of specific deep learning frameworks (e.g., `TensorFlow`, `PyTorch`, etc.).

## Getting Started

Follow the [installation guide](./docs/Installation.md).

## Diabetic Retinopathy Detection

Follow the [download and prepare guide](./docs/Benchmarks.md)
