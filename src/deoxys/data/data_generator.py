# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import numpy as np


class DataGenerator:
    def __init__(self):
        pass

    @property
    def total_batch(self):
        """
        Total number of batches to iterate all data

        : return: Total number of batches to iterate all data
        : rtype: int
        """
        return 0

    def generate(self):
        raise NotImplementedError()


class HDF5DataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None):
        if not folds or not h5file:
            raise ValueError("h5file or folds is empty")

        # Checking for existence of folds and dataset
        group_names = h5file.keys()
        dataset_names = []
        str_folds = [str(fold) for fold in folds]
        for fold in str_folds:
            if fold not in group_names:
                raise RuntimeError(
                    'HDF5 file: Fold name "{0}" is not in this h5 file'
                    .format(fold))
            if dataset_names:
                if h5file[fold].keys() != dataset_names:
                    raise RuntimeError(
                        'HDF5 file: All folds should have the same structure')
            else:
                dataset_names = h5file[fold].keys()
                if x_name not in dataset_names or y_name not in dataset_names:
                    raise RuntimeError(
                        'HDF5 file: {0} or {1} is not in the file'
                        .format(x_name, y_name))

        # Checking for valida preprocessor
        if preprocessors:
            if type(preprocessors) == list:
                for pp in preprocessors:
                    if not callable(getattr(pp, 'transform', None)):
                        raise ValueError(
                            'Preprocessor should have a "transform" method')
            else:
                if not callable(getattr(preprocessors, 'transform', None)):
                    raise ValueError(
                        'Preprocessor should have a "transform" method')

        self.hf = h5file
        self.batch_size = batch_size
        self.seg_size = batch_size * batch_cache
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name

        self.folds = str_folds

        # Cache first segment for first fold
        self.index = 0
        self.seg_index = 0

        first_fold_name = self.folds[0]

        self.x_cur = self.hf[first_fold_name][self.x_name][:self.seg_size]
        self.y_cur = self.hf[first_fold_name][self.y_name][:self.seg_size]

        # Get the total length of the first fold
        self.fold_len = len(self.hf[first_fold_name][self.y_name])

        self._total_batch = None

    @property
    def total_batch(self):
        """
        Total number of batches to iterate all data.
        It will be used as the number of steps per epochs when training or
        validating data in a model.

        : return: Total number of batches to iterate all data
        : rtype: int
        """
        if self._total_batch is None:
            total_batch = 0
            fold_names = self.folds

            for fold_name in fold_names:
                total_batch += np.ceil(
                    len(self.hf[fold_name][self.y_name]) / self.batch_size)
            self._total_batch = int(total_batch)
        return self._total_batch

    def next_fold(self):
        # Reset segment index
        self.seg_index = 0

        # Remove previous fold index and move to next one
        self.folds.append(self.folds.pop(0))

        fold_name = self.folds[0]
        y = self.hf[fold_name][self.y_name]

        # Recalculate the total length
        self.fold_len = len(y)

    def next_seg(self):
        # Reset index
        self.index = 0

        # Move segment index
        self.seg_index += self.seg_size

        # When all segments fold has been yielded, move to next fold
        if self.seg_index >= self.fold_len:
            self.next_fold()

        # store local variable after seg_index changed
        seg_index = self.seg_index

        fold_name = self.folds[0]
        # The last segment may has less items than seg_size
        if seg_index + self.seg_size >= self.fold_len:
            self.x_cur = self.hf[fold_name][self.x_name][seg_index:]
            self.y_cur = self.hf[fold_name][self.y_name][seg_index:]
        else:
            next_seg_index = seg_index + self.seg_size

            self.x_cur = self.hf[fold_name][
                self.x_name][seg_index:next_seg_index]
            self.y_cur = self.hf[fold_name][
                self.y_name][seg_index:next_seg_index]

    def generate(self):
        """
        Create a generator that generate a batch of data

        :yield: batch of (input, target)
        :rtype: tuple of 2 arrays
        """
        while True:
            # When all batches of data are yielded, move to next seg
            if self.index >= self.seg_size or \
                    self.seg_index + self.index >= self.fold_len:
                self.next_seg()

            # Index may has been reset. Thus, call after next_seg
            index = self.index

            # The last batch of data may not have less than batch_size items
            if index + self.batch_size >= self.seg_size or \
                    self.seg_index + index + self.batch_size >= self.fold_len:
                batch_x = self.x_cur[index:]
                batch_y = self.y_cur[index:]
            else:
                # Take the next batch
                batch_x = self.x_cur[index:(index + self.batch_size)]
                batch_y = self.y_cur[index:(index + self.batch_size)]

            # Apply preprocessor
            if self.preprocessors:
                if type(self.preprocessors) == list:
                    for preprocessor in self.preprocessors:
                        batch_x, batch_y = preprocessor.transform(
                            batch_x, batch_y)
                else:
                    batch_x, batch_y = self.preprocessors.transform(
                        batch_x, batch_y)

            self.index += self.batch_size
            yield batch_x, batch_y
