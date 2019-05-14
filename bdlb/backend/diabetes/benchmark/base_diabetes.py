import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json

from ..report import evaluate as evaluate
from ..report.evaluate import EvaluationResult
from ..data_loading.dataset_tf import _initialize_tf_dataset
from ...common.benchmark.base_benchmark import _BaseBenchmarkCore
from ...common.data_loading.dataset import Dataset, Dataset_PT
from ...common.report.report import Report


class _BaseDiabetes(_BaseBenchmarkCore):

  def __init__(self,
               data_path,
               train_df,
               test_df,
               augment,
               oversample,
               target_size,
               process_train,
               process_test_and_eval,
               normalize,
               data_format="channels_last",
               input_type='.jpeg'):
    self.data_path = data_path
    self.augment = augment
    self.over_sample = oversample
    self.target_size = target_size
    self.data_format = data_format
    self.normalize = normalize
    self.input_type = input_type

    self.tf_ds_train = None
    self.tf_ds_test = None
    self.tf_ds_eval = None

    train_labels = (train_df['level'].values >= 2) * 1
    train_file_names = train_df['image'].values

    n_samples = len(train_labels)
    idx_train, idx_eval = train_test_split(
        np.arange(n_samples),
        stratify=train_labels,
        test_size=0.2,
        random_state=1)

    eval_filenames = train_file_names[idx_eval]
    eval_labels = train_labels[idx_eval]

    train_labels = train_labels[idx_train]
    train_file_names = train_file_names[idx_train]

    if self.over_sample:
      pos_train_mask = train_labels == 1
      neg_train_mask = train_labels == 0
      train_file_names = np.concatenate([train_file_names[pos_train_mask]] * 4 +
                                        [train_file_names[neg_train_mask]])
      train_labels = np.array([1] * pos_train_mask.sum() * 4 +
                              [0] * neg_train_mask.sum())

    train_path = os.path.join(data_path, "train")
    self.train_ds = Dataset(
        train_path,
        target_size,
        train_file_names,
        train_labels,
        data_format,
        process_train,
        file_ext=input_type)
    self.nr_train_samples = len(train_file_names)

    self.eval_ds = Dataset(
        train_path,
        target_size,
        eval_filenames,
        eval_labels,
        data_format,
        process_test_and_eval,
        file_ext=input_type)
    self.nr_eval_samples = len(eval_filenames)

    test_labels = test_df['level'].values
    test_filenames = test_df['image'].values

    test_path = os.path.join(data_path, "test")
    self.test_ds = Dataset(
        test_path,
        target_size,
        test_filenames,
        test_labels,
        data_format,
        process_test_and_eval,
        file_ext=input_type)
    self.nr_test_samples = len(test_filenames)

  def iterate_train(self, batch_size=64, nb_workers=4, max_q_size=10):
    return self.train_ds.batch_iterator(batch_size, nb_workers, max_q_size)

  def train_tf_input_fn(self,
                        nb_workers=8,
                        batch_size=64,
                        shuffle_buffer_size=3000,
                        nr_epochs=1):

    def _input_fn():
      train_fn_input = [nb_workers, batch_size, shuffle_buffer_size, nr_epochs]
      if self.tf_ds_train is None or (
          not train_fn_input == self.train_fn_input_prev):
        path_train = os.path.join(self.data_path, 'train_tf')
        files_train = [
            os.path.join(path_train, f)
            for f in os.listdir(path_train)
            if os.path.isfile(os.path.join(path_train, f)) and
            f.endswith(".tfrecord")
        ]

        self.tf_ds_train = _initialize_tf_dataset(
            files_train,
            self.augment,
            self.over_sample,
            True,
            self.target_size,
            self.normalize,
            nb_workers,
            batch_size,
            shuffle_buffer_size,
            input_type=self.input_type,
            nr_epochs=nr_epochs)

      self.train_fn_input_prev = train_fn_input
      iterator = self.tf_ds_train.make_one_shot_iterator()

      features, labels = iterator.get_next()
      return features, labels

    return _input_fn

  def iterate_validation(self, batch_size=64, nb_workers=4, max_q_size=10):
    return self.eval_ds.batch_iterator(batch_size, nb_workers, max_q_size)

  def validate_tf_input_fn(self, nb_workers=8, batch_size=64):

    def _input_fn():
      eval_fn_input = [nb_workers, batch_size]
      if self.tf_ds_eval is None or (
          not eval_fn_input == self.eval_fn_input_prev):
        path_eval = os.path.join(self.data_path, 'eval_tf')
        files_eval = [
            os.path.join(path_eval, f)
            for f in os.listdir(path_eval)
            if os.path.isfile(os.path.join(path_eval, f)) and
            f.endswith(".tfrecord")
        ]
        self.tf_ds_eval = _initialize_tf_dataset(
            files_eval,
            False,
            False,
            False,
            self.target_size,
            self.normalize,
            nb_workers,
            batch_size,
            0,
            input_type=self.input_type)
      self.eval_fn_input_prev = eval_fn_input
      iterator = self.tf_ds_eval.make_one_shot_iterator()

      features, labels = iterator.get_next()
      return features, labels

    return _input_fn

  def get_pytorch_datasets(self, augment=False):

    import torchvision.transforms as transforms

    std = [x / 255 for x in [70.53946096, 51.71475228, 43.03428563]]
    mean = [x / 255 for x in [108.64628601, 75.86886597, 54.34005736]]

    if augment:
      target_size = 256
      padding = (int(np.random.uniform(0, 0.05) * target_size), int(np.random.uniform(0, 0.05) * target_size))
      zoom = np.random.uniform(0.9, 1.1) * target_size



      train_trsfm = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.RandomRotation(180),
          transforms.Pad(padding),
          transforms.RandomCrop(target_size, pad_if_needed=True),
          transforms.RandomCrop(zoom, pad_if_needed=True),
          transforms.Resize((target_size, target_size)),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])

      trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

    else:
      train_trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])
      trsfm = train_trsfm

    train_pt = Dataset_PT(self.train_ds, transform=train_trsfm)
    eval_pt = Dataset_PT(self.eval_ds, transform=trsfm)
    test_pt = Dataset_PT(self.test_ds, transform=trsfm)
    return train_pt, eval_pt, test_pt

  @staticmethod
  def metrics_deterministic(labels, predictions):
    acc = evaluate.accuracy(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    return EvaluationResult(acc, auc)

  def generate_report(self,
                      estimator,
                      mode='test',
                      model_class='new_model',
                      dataset_type='medium',
                      batch_size=64,
                      nb_workers=4,
                      max_q_size=10,
                      name='Test',
                      ci=False,
                      n_bootstrap=10000):
    labels = []
    predictions = []
    uncertainties = []

    if mode == 'train':
      dataset = self.train_ds
    elif mode == 'eval':
      dataset = self.eval_ds
    elif mode == 'test':
      dataset = self.test_ds
    else:
      raise ValueError("Invalid dataset supplied to generate_report, should be one of [train, eval, test]")

    n_samples = dataset.n_samples
    # i = 1
    

    with dataset.batch_iterator(batch_size, nb_workers, max_q_size) as it:
      for x, y in it:
        # print('Testing batch', i, 'out of', n_samples // batch_size)
        # i += 1

        labels.append(y)
        new_pred, new_uncert = estimator(x)
        predictions.append(new_pred)
        uncertainties.append(new_uncert)

    labels = np.concatenate(labels).flatten()
    predictions = np.concatenate(predictions).flatten()
    uncertainties = np.concatenate(uncertainties).flatten()

    return self._generate_report(ci, labels, n_bootstrap, name, predictions,
                                 uncertainties, mode, model_class, dataset_type)

  @staticmethod
  def _generate_report(ci, labels, n_bootstrap, name, predictions,
                       uncertainties, mode, model_class, dataset_type):
    labels_binary = (labels >= 2) * 1
    benchmark_model_classes = ['Deterministic_NN', 'MC_Dropout_BNN', 'MFVI_BNN']

    if not os.path.isdir('results'):
        os.mkdir('results')
        if not os.path.isdir('results/diabetes'):
            os.mkdir('diabetes')
            if not os.path.isdir('results/diabetes/{}'.format(dataset_type)):
                os.mkdir('results/diabetes/{}'.format(dataset_type))

    i = 1
    # Save model predictions and true labels to json file
    while True:
        if not os.path.isfile('results/diabetes/{}/{}_{}.json'.format(dataset_type, model_class, i)):
            data = {'labels_binary_{}_{}'.format(model_class, i): labels_binary.tolist(),
                    'predictions_{}_{}'.format(model_class, i): predictions.tolist(),
                    'uncertainties_{}_{}'.format(model_class, i): uncertainties.tolist(),
                    'new_model': '{}_{}'.format(model_class, i),
                    'model_class': model_class,
                    'dataset_type': dataset_type,
                    'model_classes': benchmark_model_classes
                    }
            if not any(model_class in s for s in data['model_classes']):
                data['model_classes'].append(model_class)
            if mode == 'test':
                with open('results/diabetes/{}/{}_{}.json'.format(dataset_type, model_class, i), 'w') as fp:
                    json.dump(data, fp)
            i = 1
            break
        else:
            i += 1

    # Load previously saved model predictions and corresponding true labels from json files and store as dict
    for m in data['model_classes']:
        while True:
            if os.path.isfile('results/diabetes/{}/{}_{}.json'.format(dataset_type, m, i)):
                with open('results/diabetes/{}/{}_{}.json'.format(dataset_type, m, i), 'r') as fp:
                    data_loaded = json.load(fp)
                data['labels_binary_{}_{}'.format(m, i)] = np.array(data_loaded['labels_binary_{}_{}'.format(m, i)])
                data['predictions_{}_{}'.format(m, i)] = np.array(data_loaded['predictions_{}_{}'.format(m, i)])
                data['uncertainties_{}_{}'.format(m, i)] = np.array(data_loaded['uncertainties_{}_{}'.format(m, i)])
                i += 1
            else:
                i = 1
                break

    acc_under_varying_ref_rate = evaluate.measure_under_varying_referral_rate(
        data, name, evaluate.accuracy, 'accuracy', ci, n_bootstrap)
    auc_under_varying_ref_rate = evaluate.measure_under_varying_referral_rate(
        data, name, roc_auc_score, 'auc', ci, n_bootstrap)
    intervals_df = evaluate.intervals_df(labels_binary, predictions, uncertainties, name)
    roc_intervals_plot = evaluate.roc_intervals_plot(labels_binary, predictions, uncertainties, name, ci, n_bootstrap)
    level_plot = evaluate.level_plot(labels, uncertainties)

    report = Report()
    report._add_figure(acc_under_varying_ref_rate, key='acc_plot')
    report._add_figure(auc_under_varying_ref_rate, key='auc_plot')
    report._add_figure(roc_intervals_plot, key='roc_plot')
    report._add_table(intervals_df, key='df')
    report._add_figure(level_plot, 'level_plot')

    perm = np.random.permutation(predictions.size)

    labels_binary_perm = labels_binary[perm]
    labels_perm = labels[perm]
    predictions_perm = predictions[perm]
    uncertainties_perm = uncertainties[perm]

    report._add_data(labels_perm, 'labels')
    report._add_data(predictions_perm, 'predictions')
    report._add_data(uncertainties_perm, 'uncertainties')
    report._add_data(labels_binary_perm, 'labels_binary')

    return report
