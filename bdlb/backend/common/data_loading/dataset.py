"""
Initial design inspired by https://github.com/chleibig/disease-detection
"""

from __future__ import print_function

import os
try:
  import Queue as queue
except ImportError:
  import queue
import numpy as np
from matplotlib.pyplot import imread
from .batch_queue import batch_queue

import torch
import torch.utils.data as data


class Dataset(object):

  def __init__(self,
               path_data,
               target_size,
               image_filenames,
               labels=None,
               data_format='channels_last',
               processing=None,
               file_ext=".jpeg"):
    """
        Object encapsulating a dataset.

        Args:
            path_data : str
                Path to data_preprocess_scripts.

            target_size : tuple
                Dimension of image in dataset.

            image_filenames : numpy array of str
                File names of images to load.

            labels : numpy array
                Labels corresponding to each file name. None if no labels present.

            data_format : str
                'channels_last' for images in (height, width, channels) format,
                or 'channels_first' for images in (channels, height, width) format.

            processing  : function numpy array image -> numpy array image
                Processing function that is called on each images,
                may include preprocessing and data_preprocess_scripts augmentation tasks.
        """
    if labels is not None:
      assert len(image_filenames) == len(labels)

    self.file_ext = file_ext
    self.path_data = path_data
    self.image_filenames = image_filenames
    self._y = labels
    self._n_samples = len(self.y)
    self.processing = processing
    self.target_size = tuple(target_size)
    self.data_format = data_format
    if self.file_ext == ".npy":
      self.image_shape = self.target_size
    elif self.data_format == 'channels_last':
      self.image_shape = self.target_size + (3,)
    elif self.data_format == 'channels_first':
      self.image_shape = (3,) + self.target_size
    else:
      raise ValueError(
          'data_format must be either channels_last or channels_first')

  @property
  def n_samples(self):
    """Number of samples in the entire dataset"""
    return self._n_samples

  @property
  def y(self):
    """Labels"""
    return self._y

  def batch_iterator(self,
                     batch_size=64,
                     nb_workers=4,
                     max_q_size=10,
                     shuffle=True):
    """
        Returns DatasetIterator over this dataset.
        Args:
            batch_size : int, 64 by default
                Size of batches.


            nb_worker : int, 4 by default
                Number of workers.

            max_q_size : int, 10 by default
                Max number of batches concurrently in the queue.

            shuffle : boolean, True by default
                If true, iterate in shuffled order.

        Returns:
            DatasetIterator with same arguments past, and with self as dataset argument.
        """
    return DatasetIterator(self, batch_size, nb_workers, max_q_size, shuffle)

  def __getitem__(self, key):
    image = self._get_images([key])[0]
    label = self._get_labels([key])[0]
    return image, label

  def __len__(self):
    return self._n_samples

  def _load_image(self, filename):
    """
        Load image.

        Args:
            filename : string
                relative filename (path to image folder gets prefixed)

        Returns:
            image : numpy array, shape according to self.data_format.
        """

    filename = os.path.join(self.path_data, filename + self.file_ext)
    if not os.path.exists(filename):
      print("missing image at {filename}".format(filename=filename))
      return None
    if self.file_ext == ".npy":
      im = np.load(filename)
    else:
      im = imread(filename)
    if self.data_format == 'channels_first':
      im = np.transpose(im, (1, 2, 0))
    return im

  def _get_images(self, indices, process=True):
    """
        Retrieves images corresponding to indices.

        Args:
            indices : array of int
                Indices to retrieve (corresponding to index in self.image_filenames)

            process : boolean
                Apply processing step.

        Returns:
            numpy array:
                4D numpy array with images. Axis 0 is image index, remaining indices
                are ordered according to self.data_format
        """
    batch = list()
    for i, file_index in enumerate(indices):
      img = self._load_image(self.image_filenames[file_index])
      if img is not None:
        if self.processing is not None and process:
          img = self.processing(img)
        batch.append(img)
    batch = np.asarray(batch)
    return batch

  def _get_labels(self, indices):
    """
        Retrieves labels corresponding to indices.

        Args:
            indices : array of int
                Indices to retrieve (corresponding to index in self.y).

        Returns:
            numpy array:
                0D array of labels.
        """
    idx = list()
    for i, file_index in enumerate(indices):
      if self._load_image(self.image_filenames[file_index]) is not None:
        idx.append(file_index)
    labels = self.y[idx]
    return labels


class DatasetIterator(object):

  def __init__(self,
               dataset,
               batch_size=64,
               nb_workers=4,
               max_q_size=10,
               multi_processing=False,
               shuffle=True):
    """
        Iterator over dataset, recommended to be used with 'with' context manager.

        Args:
            dataset : Dataset
                Dataset object to iterate over.

            batch_size : int, 64 by default
                Size of batches.


            nb_worker : int, 4 by default
                Number of workers.

            max_q_size : int, 10 by default
                Max number of batches concurrently in the queue.

            multi_processing : boolean, False by default
                If true, use multiple processes. If false, use
                multiple threads. Multiple processes are only recommended if the
                processing step of images has higher cost than IO (this is not often).

            shuffle : boolean, True by default
                If true, iterate in shuffled order.
        """
    self.dataset = dataset
    self.indices = np.arange(dataset.n_samples)
    self.batch_size = batch_size
    self.nb_workers = nb_workers
    self.max_q_size = max_q_size
    self.multi_processing = multi_processing
    if shuffle:
      np.random.shuffle(self.indices)
    self.batch_index = 0
    self.nr_of_batches = len(self.indices) // self.batch_size

    # Initialize local variables for managing the BatchQueue.
    self.batchq = None
    self.q = None
    self.stop = None

  def __iter__(self):
    return self

  def __enter__(self):
    self.q, self.stop = batch_queue(
        self.dataset,
        self.indices,
        self.batch_size,
        max_q_size=self.max_q_size,
        nb_worker=self.nb_workers)
    return self

  def __exit__(self, *args):
    self.stop.set()

  def __next__(self):
    if self.q is None or self.stop is None:
      raise ValueError(
          "Must call __enter__ method on iterator before next. Recommend"
          " using 'with' context manager for iteration.")
    if self.batch_index < self.nr_of_batches:
      while True:
        try:
          batch = self.q.get(block=True, timeout=0.1)
          break
        except queue.Empty:
          continue
      self.batch_index += 1
      return batch
    else:
      self.stop.set()
      raise StopIteration()

  next = __next__


class Dataset_PT(data.Dataset):
    """
    Class wraps Dataset and provides functions for pytorch dataloader.
    """
    def __init__(self, dataset, transform=None):
        """

        :param dataset: a data.Dataset object
        :param transform: a composition of torchvision.transforms functions
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, key):
      """
      For dataloader
      :param key: index of the example in the dataset
      :return: torch.Tensor with shape [channels, height, width], torch.Tensor []
      """
      image = self.dataset._load_image(self.dataset.image_filenames[key])
      label = self.dataset.y[key]

      image = self.transform(np.uint8(image))


      return image, label

    def __len__(self):
      return self.dataset.n_samples