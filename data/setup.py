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

import os
import zipfile
import subprocess


class Diabetes:
  """Helper class for diabetic retinopathy detection dataset."""

  def __init__(self, DATADIR="./data"):
    """Initializes an object.

    Args:
      DATADIR: String, relative path to the parent directory
        to store the data for the tasks.
    """
    self.DATADIR = DATADIR
    self.PARDIR = os.path.join(self.DATADIR, "diabetes")
    self.RAW = os.path.join(self.PARDIR, "raw")

  def run(self):
    """Run setup command."""
    # Unzip DiabetesToy experiment hosted data
    self.fetch_toy_n_splits()
    # Fetch raw data from Kaggle
    self.download_from_kaggle()
    # Unzip files and remove archives
    self.unzip()
    # Preprocess data
    self.preprocess()

  def fetch_toy_n_splits(self):
    """Setup toy task and split CSVs."""
    HOSTED = os.path.join(self.RAW, "diabetes.zip")
    # Download if not there
    if not os.path.exists(HOSTED):
      subprocess.check_call([
          "wget", "-P", self.RAW,
          "http://oatml.cs.ox.ac.uk/benchmarks/diabetes.zip"
      ])
    # Unzip archibe
    with zipfile.ZipFile(HOSTED) as zip_:
      zip_.extractall(self.DATADIR)
    # Delete .zip
    os.remove(HOSTED)

  def download_from_kaggle(self):
    """Download data from Kaggle."""
    # Check if "kaggle" is globally available
    if subprocess.call(["which", "kaggle"]) == 0:
      KAGGLE = "kaggle"
    # Use "~/.local/bin/kaggle"
    else:
      KAGGLE = "../.local/bin/kaggle"
    command = [
        KAGGLE, "competitions", "download", "-c",
        "diabetic-retinopathy-detection", "-p", self.RAW
    ]
    subprocess.check_call(command)

  def unzip(self):
    """Unzip raw data."""
    # Concatenate "train.zip.00*" to "train.zip"
    train_batch = [
        "7z", "x",
        os.path.join(self.RAW, "train.zip.001"), "-tsplit",
        "-o{}".format(self.RAW)
    ]
    subprocess.check_call(train_batch)
    # Delete "train.zip.00*" files
    for trainzip in os.listdir(self.RAW):
      if "train.zip.00" in trainzip:
        os.remove(os.path.join(self.RAW, trainzip))
    # Unzip "train.zip"
    subprocess.check_call([
        "7z", "x",
        os.path.join(self.RAW, "train.zip"), "-o{}".format(self.RAW)
    ])
    # Delete "train.zip"
    os.remove(os.path.join(self.RAW, "train.zip"))
    # Concatenate "test.zip.00*" to "test.zip"
    test_batch = [
        "7z", "x",
        os.path.join(self.RAW, "test.zip.001"), "-tsplit",
        "-o{}".format(self.RAW)
    ]
    subprocess.check_call(test_batch)
    # Delete "test.zip.00*"
    for testzip in os.listdir(self.RAW):
      if "test.zip.00" in testzip:
        os.remove(os.path.join(self.RAW, testzip))
    # Unzip "test.zip"
    subprocess.check_call([
        "7z", "x",
        os.path.join(self.RAW, "test.zip"), "-o{}".format(self.RAW)
    ])
    # Delete "test.zip"
    os.remove(os.path.join(self.RAW, "test.zip"))
    # Delete split files
    for splitfname in [
        "sample.zip", "sampleSubmission.csv.zip", "trainLabels.csv.zip"
    ]:
      os.remove(os.path.join(self.RAW, splitfname))

  def preprocess(self):
    """Preprocesses the raw data from Kaggle.
    Image conversion: crop, resize, (enhance
    colour contrast) and save.
    """
    import sys
    import math
    import shutil
    import numpy as np
    from PIL import Image
    from tqdm import trange
    import multiprocess.pool
    from bdlb.tasks import DiabetesRealWorld, DiabetesMedium
    white_list_extensions = ["jpg", "jpeg", "JPEG", "tif"]
    sys.path.append(os.getcwd())

    def _int64_feature(value):
      import tensorflow as tf
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
      import tensorflow as tf
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to_tf_record(ds, path, nr_shards, name):
      import tensorflow as tf
      shard_size = int(math.ceil(ds.image_filenames.size // nr_shards))
      if not os.path.exists(path):
        os.makedirs(path)
      perm = np.random.RandomState(seed=42).permutation(ds.image_filenames.size)
      for shard_nr in range(nr_shards):
        file_name = os.path.join(path, "{}_{}.tfrecord".format(name, shard_nr))
        with tf.python_io.TFRecordWriter(file_name) as writer:
          for i in range(
              shard_size * shard_nr,
              min(shard_size * (shard_nr + 1), ds.image_filenames.size)):
            # write label, shape, and image content to the TFRecord file
            idx = perm[i]
            img = ds._get_images(
                np.array([idx]), process=False).astype(np.uint8)
            label = ds._get_labels(np.array([idx])).reshape(())
            shape = list(img.shape)
            shape = shape[1:]
            img = img.reshape(tuple(shape))
            shape = np.array(shape)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": _int64_feature(label),
                        "shape": _bytes_feature(shape.tobytes()),
                        "image": _bytes_feature(img.tobytes())
                    }))
            writer.write(example.SerializeToString())
            if (i + 1) % 100 == 0:
              print("Progress:",
                    "%.1f%% " % (100 * i / ds.image_filenames.size))

    def convert(fname, crop_size):
      """Refactored from JF"s generators.load_image_and_process"""
      im = Image.open(fname, mode="r")

      assert len(np.shape(im)) == 3, "Shape of image {} unexpected, " \
          "maybe it's grayscale".format(fname)

      w, h = im.size

      if w / float(h) >= 1.3:
        cols_thres = np.where(
            np.max(np.max(np.asarray(im), axis=2), axis=0) > 35)[0]

        # Extra cond compared to orig crop.
        if len(cols_thres) > crop_size // 2:
          min_x, max_x = cols_thres[0], cols_thres[-1]
        else:
          min_x, max_x = 0, -1

        converted = im.crop((min_x, 0, max_x, h))

      else:  # no crop
        converted = im

      # Resize without preserving aspect ratio:
      converted = converted.resize((crop_size, crop_size),
                                   resample=Image.BILINEAR)
      return converted

    def get_convert_fname(fname, extension, directory, convert_directory):
      source_extension = fname.split(".")[-1]
      fname = fname.replace(source_extension, extension)
      return os.path.join(convert_directory, os.path.basename(fname))

    def create_dirs(paths):
      for p in paths:
        try:
          os.makedirs(p)
        except OSError:
          pass

    def process(args):
      fun, arg = args
      directory, convert_directory, fname, crop_size, \
          extension = arg
      convert_fname = get_convert_fname(fname, extension, directory,
                                        convert_directory)
      if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        # print(convert_fname)
        if img is not None:
          save(img, convert_fname)

    def save(img, fname):
      img.save(fname, quality=97)

    # Create folder structure
    os.makedirs(os.path.join(self.PARDIR, "realworld"), exist_ok=True)
    os.makedirs(os.path.join(self.PARDIR, "realworld", "train"), exist_ok=True)
    os.makedirs(os.path.join(self.PARDIR, "realworld", "test"), exist_ok=True)
    os.makedirs(os.path.join(self.PARDIR, "medium"), exist_ok=True)
    os.makedirs(os.path.join(self.PARDIR, "medium", "train"), exist_ok=True)
    os.makedirs(os.path.join(self.PARDIR, "medium", "test"), exist_ok=True)

    extension = "jpeg"

    filenames_test = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(self.RAW)
        for f in fn
        if (f.split(".")[-1] in white_list_extensions and
            os.path.split(dp)[-1] == "test")
    ]

    filenames_train = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(self.RAW)
        for f in fn
        if (f.split(".")[-1] in white_list_extensions and
            os.path.split(dp)[-1] == "train")
    ]

    assert filenames_test and filenames_train, "Error finding valid filenames."

    print("Resizing images in {} to {}, this takes a while."
          "".format(self.RAW, self.PARDIR))

    n = len(filenames_test) + len(filenames_train)
    batchsize = 500

    args = []

    for f in filenames_train:
      args.append((convert, (self.RAW,
                             os.path.join(self.PARDIR, "realworld", "train"), f,
                             512, extension)))
      args.append((convert, (self.RAW,
                             os.path.join(self.PARDIR, "medium", "train"), f,
                             256, extension)))

    for f in filenames_test:
      args.append((convert, (self.RAW,
                             os.path.join(self.PARDIR, "realworld", "test"), f,
                             512, extension)))
      args.append((convert, (self.RAW,
                             os.path.join(self.PARDIR, "medium", "test"), f,
                             256, extension)))

    batches = (len(args) // batchsize) + 1

    with multiprocess.pool.Pool(processes=None) as pool:
      for i in trange(batches):
        pool.map(process, args[i * batchsize:(i + 1) * batchsize])

    data_sets = {
        'realworld': DiabetesRealWorld(self.PARDIR),
        'medium': DiabetesMedium(self.PARDIR)
    }

    for k, v in data_sets.items():
      print('Generating TFRecords training set for', k)
      train_ds = v.train_ds
      convert_to_tf_record(train_ds, os.path.join(self.PARDIR, k, 'train_tf'),
                           8, 'train')

      print('Generating TFRecords evaluation set for', k)
      eval_ds = v.eval_ds
      convert_to_tf_record(eval_ds, os.path.join(self.PARDIR, k, 'eval_tf'), 8,
                           'eval')

    # Delete "raw" folder
    shutil.rmtree(self.RAW)


if __name__ == "__main__":
  import os
  import argparse
  # CLI argument parser object
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-b",
      "--benchmark",
      type=str,
      help="Benchmark to setup (diabetes | segmentation).")
  parser.add_argument(
      "-u",
      "--kaggle-user",
      type=str,
      help="<Optional> Kaggle username for fetching data from API.")
  parser.add_argument(
      "-k",
      "--kaggle-key",
      type=str,
      help="<Optional> Kaggle password for fetching data from API.")
  parser.add_argument(
      "-d",
      "--data-dir",
      type=str,
      help="<Optional> Root data directory to store the data.")
  # Parse and process options
  argv = parser.parse_args()
  DATADIR = argv.data_dir or "./data"
  # === Diabetic Retinopathy Detection ===
  if argv.benchmark == "diabetes":
    # Setup Kaggle credentials
    kaggle_user = argv.kaggle_user or os.getenv("KAGGLE_USERNAME")
    kaggle_key = argv.kaggle_key or os.getenv("KAGGLE_KEY")
    if kaggle_user is None:
      raise IOError(
          "Neither option --kaggle-user, nor the environment variable KAGGLE_USERNAME was found."
      )
    else:
      # Override environment variable "KAGGLE_USERNAME"
      os.environ["KAGGLE_USERNAME"] = kaggle_user
    if kaggle_key is None:
      raise IOError(
          "Neither option --key, nor the environment variable KAGGLE_KEY was found."
      )
    else:
      # Override environment variable "KAGGLE_USERNAME"
      os.environ["KAGGLE_KEY"] = kaggle_key
    # Create runner
    diabetes = Diabetes(DATADIR=DATADIR)
    # Run command
    diabetes.run()
  # === Cityscapes Segmentation ===
  elif argv.benchmark == "cityscapes":
    pass
  else:
    raise ValueError(
        "The provided option --benchmark={benchmark} is currently not supported."
        "Use instead one of ('diabetes', 'cityscapes').".format(
            benchmark=argv.benchmark))
