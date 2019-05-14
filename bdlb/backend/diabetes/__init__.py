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

import numpy as np
import pandas as pd

from .benchmark.base_diabetes import _BaseDiabetes
from .data_loading.image_preprocess import image_preprocess


class DiabetesMedium(_BaseDiabetes):

  def __init__(self, path="data/diabetes", data_format="channels_last", process_train=True, process_test_and_eval=True):
    data_path = os.path.join(path, "medium")
    train_df = pd.read_csv(
        os.path.join(path, "trainLabelsSmall.csv"), dtype={'level': np.int32})
    test_df = pd.read_csv(
        os.path.join(path, "testLabelsSmall.csv"), dtype={'level': np.int32})

    if process_train:
        process_train = image_preprocess(data_format)
    else:
        process_train = None
    if process_test_and_eval:
        process_test_and_eval = image_preprocess(data_format, False)
    else:
        process_test_and_eval = None

    super(DiabetesMedium, self).__init__(
        data_path=data_path,
        train_df=train_df,
        test_df=test_df,
        augment=True,
        oversample=False,
        target_size=(256, 256),
        process_train=process_train,
        process_test_and_eval=process_test_and_eval,
        normalize=True,
        data_format=data_format)


class DiabetesRealWorld(_BaseDiabetes):

  def __init__(self, path="data/diabetes", data_format="channels_last"):
    data_path = os.path.join(path, "realworld")
    train_df = pd.read_csv(
        os.path.join(path, "trainLabels.csv"), dtype={'level': np.int32})
    test_df = pd.read_csv(
        os.path.join(path, "testLabels.csv"), dtype={'level': np.int32})

    process_train = image_preprocess(data_format)
    process_test_and_eval = image_preprocess(data_format, False)

    super(DiabetesRealWorld, self).__init__(
        data_path=data_path,
        train_df=train_df,
        test_df=test_df,
        augment=True,
        oversample=True,
        target_size=(512, 512),
        process_train=process_train,
        process_test_and_eval=process_test_and_eval,
        normalize=True,
        data_format=data_format)


class DiabetesToy(_BaseDiabetes):

  def __init__(self, path="data/diabetes"):
    data_path = os.path.join(path, "toy")
    if not os.path.exists(data_path):
      print("Downloading Toy dataset from server")
      import urllib.request
      import zipfile
      import shutil
      URL = "http://oatml.cs.ox.ac.uk/benchmarks/diabetes.zip"
      FOLDER = "data/"
      FNAME = "diabetes.zip"
      # Fetch hosted data
      pwd = os.path.abspath("../../")
      try:
        os.mkdir(os.path.join(pwd, FOLDER))
      except OSError:
        pass
      with open(os.path.join(pwd, FOLDER, FNAME), "wb") as dfile:
        dfile.write(urllib.request.urlopen(URL).read())
      # Extract zip file
      with zipfile.ZipFile(os.path.join(pwd, FOLDER, FNAME)) as zfile:
        zfile.extractall(os.path.join(pwd, FOLDER))
      # Delete zip file
      os.remove(os.path.join(pwd, FOLDER, FNAME))
      # Delete __MACOSX folder
      if os.path.exists(os.path.join(pwd, FOLDER, "__MACOSX")):
        shutil.rmtree(os.path.join(pwd, FOLDER, "__MACOSX"))
    train_df = pd.read_csv(
        os.path.join(path, "trainLabelsSmall.csv"), dtype={'level': np.int32})
    test_df = pd.read_csv(
        os.path.join(path, "testLabelsSmall.csv"), dtype={'level': np.int32})

    process_train = None
    process_test_and_eval = None

    super(DiabetesToy, self).__init__(
        data_path=data_path,
        train_df=train_df,
        test_df=test_df,
        augment=False,
        oversample=False,
        target_size=(512,),
        process_train=process_train,
        process_test_and_eval=process_test_and_eval,
        normalize=False,
        data_format="channels_last",
        input_type=".npy")
