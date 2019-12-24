# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .data_generator import DataGenerator, HDF5DataGenerator
from ..utils import Singleton


class DataReader:
    """
    The base class of the Data Reader. Any newly created DataReader will
    inherit from this class.
    """

    def __init__(self, filename, batch_size=32, preprocessors=None, **kwargs):
        pass

    @property
    def train_generator(self):
        """
        Data Generator for the training dataset

        :return: an DataGenerator instance that generates the train dataset
        :rtype: deoxys.data.DataGenerator
        """
        return DataGenerator().generate()

    @property
    def test_generator(self):
        """
        Data Generator for the test dataset

        :return: an DataGenerator instance that generates the test dataset
        :rtype: deoxys.data.DataGenerator
        """
        return DataGenerator().generate()

    @property
    def val_generator(self):
        """
        Data Generator for the validation dataset

        :return: an DataGenerator instance that generates the
        validataion dataset
        :rtype: deoxys.data.DataGenerator
        """
        return DataGenerator().generate()

    @property
    def original_test(self):
        pass


class HDF5Reader(DataReader):
    """
    DataReader that use data from an hdf5 file.
    """

    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold'):
        """
        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.

        Example:

        The dataset X contain 1000 samples, with 4 columns:
        x, y, z, t. Where x is the main input, y and z are supporting
        information (index, descriptions) and t is the target for
        prediction. We want to test 30% of this dataset, and have a
        cross validation of 100 samples.

        Then, the hdf5 containing dataset X should have 10 groups,
        each group contains 100 samples. We can name these groups
        'fold_1', 'fold_2', 'fold_3', ... , 'fold_9', 'fold_10'.
        Each group will then have 4 datasets: x, y, z and t, each of
        which has 100 items.

        Since x is the main input, then `x_name='x'`, and t is the
        target for prediction, then `y_name='t'`. We named the groups
        in the form of fold_n, then `fold_prefix='fold'`.

        Let's assume the data is stratified, we want to test on the
        last 30% of the data, so `test_folds=[8, 9, 10]`.
        100 samples is used for cross-validation. Thus, one option for
        `train_folds` and `val_folds` is `train_folds=[1,2,3,4,5,6]`
        and `val_folds=[7]`. Or in another experiment, you can set
        `train_folds=[2,3,4,5,6,7]` and `val_folds=[1]`.

        If the hdf5 didn't has any formular for group name, then you
        can set `fold_prefix=None` then put the full group name
        directly to `train_folds`, `val_folds` and `test_folds`.

        :param filename: the hdf5 file name that contains the data.
        :type filename: str
        :param batch_size: number of sample to feeds in
        the neural network in each step, defaults to 32
        :type batch_size: int, optional
        :param preprocessors: list of preprocessors to apply on the data,
        defaults to None
        :type preprocessors: list of deoxys.data.Preprocessor, optional
        :param x_name: dataset name to be use as input, defaults to 'x'
        :type x_name: str, optional
        :param y_name: dataset name to be use as target, defaults to 'y'
        :type y_name: str, optional
        :param batch_cache: number of batches to be cached when reading the
        file, defaults to 10
        :type batch_cache: int, optional
        :param train_folds: list of folds to be use as train data,
        defaults to None
        :type train_folds: list of int, or list of str, optional
        :param test_folds: list of folds to be use as test data,
        defaults to None
        :type test_folds: list of int, or list of str, optional
        :param val_folds: list of folds to be use as validation data,
        defaults to None
        :type val_folds: list of int, or list of str, optional
        :param fold_prefix: the prefix of the group name,
        defaults to 'fold'
        :type fold_prefix: str, optional
        """

        self.hf = h5py.File(filename, 'r')
        self.batch_size = batch_size
        self.batch_cache = batch_cache
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        train_folds = list(train_folds) if train_folds else [0]
        test_folds = list(test_folds) if test_folds else [2]
        val_folds = list(val_folds) if val_folds else [1]

        if fold_prefix:
            self.train_folds = ['{}_{}'.format(
                fold_prefix, train_fold) for train_fold in train_folds]
            self.test_folds = ['{}_{}'.format(
                fold_prefix, test_fold) for test_fold in test_folds]
            self.val_folds = ['{}_{}'.format(
                fold_prefix, val_fold) for val_fold in val_folds]
        else:
            self.train_folds = train_folds
            self.test_folds = test_folds
            self.val_folds = val_folds

        self._original_test = None
        self._original_val = None

    @property
    def train_generator(self):
        """
        :return: A DataGenerator for generating batches of data for training
        :rtype: deoxys.data.DataGenerator
        """
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds)

    @property
    def test_generator(self):
        """
        :return: A DataGenerator for generating batches of data for testing
        :rtype: deoxys.data.DataGenerator
        """
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds)

    @property
    def val_generator(self):
        """
        :return: A DataGenerator for generating batches of data for validation
        :rtype: deoxys.data.DataGenerator
        """
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds)

    @property
    def original_test(self):
        """
        Return a dictionary of all data in the test set
        """
        if self._original_test is None:
            self._original_test = {}
            for key in self.hf[self.test_folds[0]].keys():
                data = None
                for fold in self.test_folds:
                    new_data = self.hf[fold][key][:]

                    if data is None:
                        data = new_data
                    else:
                        data = np.concatenate((data, new_data))
                self._original_test[key] = data

        return self._original_test

    @property
    def original_val(self):
        """
        Return a dictionary of all data in the val set
        """
        if self._original_val is None:
            self._original_val = {}
            for key in self.hf[self.val_folds[0]].keys():
                data = None
                for fold in self.val_folds:
                    new_data = self.hf[fold][key][:]

                    if data is None:
                        data = new_data
                    else:
                        data = np.concatenate((data, new_data))
                self._original_val[key] = data

        return self._original_val


class DataReaders(metaclass=Singleton):
    """
    A singleton that contains all the registered customized DataReaders
    """

    def __init__(self):
        self._dataReaders = {
            'HDF5Reader': HDF5Reader
        }

    def register(self, key, preprocessor):
        if not issubclass(preprocessor, DataReader):
            raise ValueError(
                "The customized preprocessor has to be a subclass"
                + " of deoxys.data.DataReader"
            )

        if key in self._dataReaders:
            raise KeyError(
                "Duplicated key, please use another key for this preprocessor"
            )
        else:
            self._dataReaders[key] = preprocessor

    def unregister(self, key):
        if key in self._dataReaders:
            del self._dataReaders[key]

    @property
    def data_readers(self):
        return self._dataReaders


def register_datareader(key, preprocessor):
    """
    Register the customized preprocessor.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the preprocessor
    :type key: str
    :param preprocessor: the customized preprocessor class
    :type preprocessor: deoxys.data.DataReader
    """
    DataReaders().register(key, preprocessor)


def unregister_datareader(key):
    """
    Remove the registered preprocessor with the key-name

    :param key: the key-name of the preprocessor to be removed
    :type key: str
    """
    DataReaders().unregister(key)


def _deserialize(config, custom_objects={}):
    return custom_objects[config['class_name']](**config['config'])


def datareader_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define preprocessor')

    if 'config' not in config:
        # auto add empty config for preprocessor with only class_name
        config['config'] = {}
    return _deserialize(config, custom_objects=DataReaders().data_readers)
