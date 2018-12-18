#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import gzip #Biblioteka rozpakowuje cyfry spakowane do formatu .gz
# Przy tak niewielkiej próbce (150) raczej nie będę musiał ich pakować,
# więc część kodu z funkcji zapewne też odpadnie.

import os #Ta biblioteka mi się tym razem nie przyda.
# os używany jest tu tylko do tworzenia ścieżki po automatycznym pobraniu
# gotowych danych z internetu. Ja nie będę hostował na razie danych u siebie,
# tylko wstawię je i sformatuję bezpośrednio do folderu. 
from six.moves.urllib.request import urlretrieve #J.W.

import numpy # Służy do czytania danych, których potem karmimy tensora.
# Bez numpyego się nie obejdzie. Alternatuwą jest pandas, który również go zawiera.

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/' #Źródło danych - u mnie do zmiany.


def maybe_download(filename, work_directory): #Ta funkcja u mnie odpadnie.
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream): #Funkcja występuje przy ekstrakcji obrazków i ich opisów.
    dt = numpy.dtype(numpy.uint32).newbyteorder('>') #Tu wygląda na to, że następuje szeregowanie bajtów.
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0] #To ma sens w kontekście #1#

def extract_images(filename): #Wyciąga obrazki z plików .gz i przyporządkowuje je
    # do 4-wymiarowej matrycy o strukturze [Lp, Y, X, nasycenie(Alpha)]
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename) #Ta linijka do zmiany, bo nie rozpakowujemy.
    with gzip.open(filename) as bytestream: #Ta linijka odpadnie. 
        magic = _read32(bytestream) # Dowiedz się po co ten error check
        if magic != 2051:           # była gdzieś jakaś liczba przy MNIST
            raise ValueError(       # i on ją tu sprawdza, ale dlaczego taka?
                'Invalid magic number %d in MNIST image file: %s' % # Możliwe, że u mnie nie będzie potrzebne, ale lepiej wiedzieć ocb.
                (magic, filename))  
        num_images = _read32(bytestream) #Tutaj wyciągamy te cztery wymiary
        rows = _read32(bytestream)       #jako zmienne przy pomocy funkcji _read32
        cols = _read32(bytestream)       #tj. mamy trzy zmienne, bo natężenie (tutaj jako cyfry) docelowo występuje w formie powiedzmy [[[0,1],[2,1]],[[0,5],[8,1]]] x Lp
        buf = bytestream.read(rows * cols * num_images) #1# Tutaj byśmy składali takie trzy strumienie sformatowane identycznie w jeden "ciąg" wciąż o formacie 32. On to nazywa buforem.
        data = numpy.frombuffer(buf, dtype=numpy.uint8) #A tutaj ten 'bufor' konwertujemy na nowy typ danych 'uint8'. Z tego co o nim wiem zapewne chodzi o kodowanie w skali szarości (Alpha).
        data = data.reshape(num_images, rows, cols, 1) #Też metoda numpy'ego. Zwraca ciąg z bufora 'buf' w nowym kształcie bez zmiany danych. Zapewne po to, aby TF mógł to potem odczytać.
        return data # Dane uszeregowane w jeden ciąg. Chyba. Zobaczmy co potem korzysta z "data"?
                    # Poza analizą kodu zobaczmy jeszcze tutorial krok po kroku do porównania.

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images # Definiuje '_właściwości', żeby się nie nadpisywały.

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
