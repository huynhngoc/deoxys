# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .data_generator import DataGenerator, HDF5DataGenerator
from .preprocessor import WindowingPreprocessor
from ..utils import Singleton


class DataReader:
    def __init__(self, filename, batch_size=32, preprocessors=None, **kwargs):
        pass

    @property
    def train_generator(self):
        return DataGenerator().generate()

    @property
    def test_generator(self):
        return DataGenerator().generate()

    @property
    def val_generator(self):
        return DataGenerator().generate()

    @property
    def original_test(self):
        pass


class HDF5Reader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold'):
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

    @property
    def train_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds)

    @property
    def test_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds)

    @property
    def val_generator(self):
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

                    if data:
                        data = np.concatenate((data, new_data))
                    else:
                        data = new_data
                self._original_test[key] = data

        return self._original_test


class KerasImageDataGenerator:
    def __init__(self, img_datagen, x_train, y_train, x_test, y_test):
        if type(img_datagen) is not ImageDataGenerator:
            raise ValueError("This data reader requires an instance from "
                             "from tensorflow.keras.preprocessing.image."
                             "ImageDataGenerator")
        self.img_datagen = img_datagen
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class DataReaders(metaclass=Singleton):
    """
    A singleton that contains all the registered customized preprocessors
    """

    def __init__(self):
        self._dataReaders = {
            'HDF5Reader': HDF5Reader,
            'KerasImageDataGenerator': KerasImageDataGenerator
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
