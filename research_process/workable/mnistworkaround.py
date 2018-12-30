
# coding: utf-8

# In[ ]:


# W komentarzu poniżej oryginalna licencja. 
# Jest to standardowa licencja Apache 2.0., która dodatkowo mówi nam ni mniej ni więcej tyle, 
# że twórcy TF nie odpowiadają za użytkowanie przez nas tego obejścia.
# Wszelkie zmiany do tego kodu wymuszone były na mnie tym, że inaczej nie działał. ;)
# Komentarze w języku polskim dla lepszego zrozumienia działania mechanizmu, jeśli się za to wezmę.

#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy as np
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

import collections
import csv
import os
from os import path
import random
import tempfile
import time

from six.moves import urllib

from tensorflow.contrib.framework import deprecated
from tensorflow.python.platform import gfile

# W tym miejscu definiujemy inne źródło danych MNIST. Strona Yann w komentarzu.
# OStatnio jak sprawdzałem działała.
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1):
  """Load dataset from CSV file with a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    n_samples = int(header[0])
    n_features = int(header[1])
    data = np.zeros((n_samples, n_features), dtype=features_dtype)
    target = np.zeros((n_samples,), dtype=target_dtype)
    for i, row in enumerate(data_file):
      target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
      data[i] = np.asarray(row, dtype=features_dtype)

  return Dataset(data=data, target=target)


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)


def shrink_csv(filename, ratio):
  """Create a smaller dataset of only 1/ratio of original data."""
  filename_small = filename.replace('.', '_small.')
  with gfile.Open(filename_small, 'w') as csv_file_small:
    writer = csv.writer(csv_file_small)
    with gfile.Open(filename) as csv_file:
      reader = csv.reader(csv_file)
      i = 0
      for row in reader:
        if i % ratio == 0:
          writer.writerow(row)
        i += 1


def load_iris(data_path=None):
  """Load Iris dataset.
  Args:
      data_path: string, path to iris dataset (optional)
  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'iris.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.int,
      features_dtype=np.float)


def load_boston(data_path=None):
  """Load Boston housing dataset.
  Args:
      data_path: string, path to boston dataset (optional)
  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'boston_house_prices.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.float,
      features_dtype=np.float)


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
  """Simple decorator for wrapping retriable functions.
  Args:
    initial_delay: the initial delay.
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    max_delay: the maximum delay allowed (actual max is
        max_delay * (1 + jitter).
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.
  """
  if factor < 1:
    raise ValueError('factor must be >= 1; was %f' % (factor,))

  if jitter >= 1:
    raise ValueError('jitter must be < 1; was %f' % (jitter,))

  # Generator to compute the individual delays
  def delays():
    delay = initial_delay
    while delay <= max_delay:
      yield delay * random.uniform(1 - jitter,  1 + jitter)
      delay *= factor

  def wrap(fn):
    """Wrapper function factory invoked by decorator magic."""

    def wrapped_fn(*args, **kwargs):
      """The actual wrapper function that applies the retry logic."""
      for delay in delays():
        try:
          return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except)
          if is_retriable is None:
            continue

          if is_retriable(e):
            time.sleep(delay)
          else:
            raise
      return fn(*args, **kwargs)
    return wrapped_fn
  return wrap


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
  return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
  return urllib.request.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    temp_file_name, _ = urlretrieve_with_retry(source_url)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = maybe_download(TRAIN_IMAGES, train_dir,
                                   source_url + TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = maybe_download(TRAIN_LABELS, train_dir,
                                   source_url + TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir,
                                   source_url + TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = maybe_download(TEST_LABELS, train_dir,
                                   source_url + TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                     .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)

