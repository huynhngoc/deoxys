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
        return DataGenerator().generate()

    @property
    def test_generator(self):
        return DataGenerator().generate()

    @property
    def val_generator(self):
        return DataGenerator().generate


class HDF5Reader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y',
                 train_folds=[0], test_folds=[1], val_folds=[2],
                 fold_prefix='fold'):
        self.hf = h5py.File(filename, 'r')
        self.batch_size = batch_size,
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        self.train_folds = list(train_folds)
        self.test_folds = list(test_folds)
        self.val_folds = list(val_folds)

    @property
    def train_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.train_folds)

    @property
    def test_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.test_folds)

    @property
    def val_generator(self):
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            fold_prefix=self.fold_prefix, folds=self.val_folds)
