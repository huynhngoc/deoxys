# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import h5py
from .data_generator import DataGenerator, HDF5DataGenerator


class DataReader:
    def __init__(self, filename, batch_size=32, preprocessors=None, **kwargs):
        pass

    @property
    def train_generator(self):
        return DataGenerator()

    @property
    def test_generator(self):
        return DataGenerator()


class HDF5Reader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y',
                 total_folds=2, fold_prefix='fold', test_fold_idx=None):
        self.hf = h5py.File(filename, 'r')
        self.batch_size = batch_size,
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name
        self.total_folds = total_folds
        self.fold_prefix = fold_prefix

        # List of test fold indice
        if test_fold_idx:
            self._test_folds = [test_fold_idx] if type(
                test_fold_idx) == int else list(test_fold_idx)
        else:
            self._test_folds = [total_folds - 1]

        # List of train folds indice
        self._train_folds = [i for i in range(
            total_folds) if i not in self._test_folds]

    @property
    def train_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self._train_folds).generate()

    def test_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self._test_folds).generate()
