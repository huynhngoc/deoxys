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
        return 0

    def generate(self):
        raise NotImplementedError()


class HDF5DataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', fold_prefix='fold', folds=[]):
        self.hf = h5file
        self.batch_size = batch_size
        self.seg_size = batch_size * batch_cache
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix
        self.folds = list(folds)

        # Cache first segment for first fold
        self.index = 0
        self.seg_index = 0

        first_fold_name = '{}_{}'.format(self.fold_prefix, self.folds[0])

        self.x_cur = self.hf[first_fold_name][self.x_name][:self.seg_size]
        self.y_cur = self.hf[first_fold_name][self.y_name][:self.seg_size]

        # Get the total length of the first fold
        self.fold_len = len(self.hf[first_fold_name][self.y_name])

        self._total_batch = None

    @property
    def total_batch(self):
        if self._total_batch is None:
            total_batch = 0
            fold_names = ['{}_{}'.format(
                self.fold_prefix, fold) for fold in self.folds]

            for fold_name in fold_names:
                total_batch += np.ceil(
                    len(self.hf[fold_name][self.y_name]) / self.batch_size)
            self._total_batch = total_batch
        return self._total_batch

    def next_fold(self):
        # Reset segment index
        self.seg_index = 0

        # Remove previous fold index and move to next one
        self.folds.append(self.folds.pop(0))

        fold_name = '{}{}'.format(self.fold_prefix, self.folds[0])
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

        fold_name = '{}_{}'.format(self.fold_prefix, self.folds[0])
        # The last segment may has less items then seg_size
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
        while True:
            # When all batches of data are yielded, move to next fold
            if self.index >= self.batch_size:
                self.next_seg()

            # Index may has been reset. Thus, call after next_seg
            index = self.index

            # The last batch of data may not have less than batch_size items
            if index + self.batch_size >= self.seg_size or \
                    index + self.batch_size >= self.fold_len:
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
