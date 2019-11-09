# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import h5py
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
        return DataGenerator().generate


class HDF5Reader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold'):
        self.hf = h5py.File(filename, 'r')
        self.batch_size = batch_size
        self.batch_cache = batch_cache
        self.preprocessors = preprocessors or [
            WindowingPreprocessor(70 + 1024, 200, 0)]
        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        self.train_folds = list(train_folds) if train_folds else [0]
        self.test_folds = list(test_folds) if test_folds else [2]
        self.val_folds = list(val_folds) if val_folds else [1]

    @property
    def train_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.train_folds)

    @property
    def test_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.test_folds)

    @property
    def val_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.val_folds)


class DataReaders(metaclass=Singleton):
    """
    A singleton that contains all the registered customized preprocessors
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
    def preprocessors(self):
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
    return _deserialize(config, custom_objects=DataReaders().preprocessors)
