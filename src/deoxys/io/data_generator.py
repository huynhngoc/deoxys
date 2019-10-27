# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


class DataGenerator:
    def __init__(self):
        pass

    def generate(self):
        raise NotImplementedError()


class HDF5DataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', fold_prefix='fold', folds=[]):
        self.hf = h5file
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix
        self.folds = list(folds)

        # Cache first fold
        self.index = 0
        self.x_cur = self.hf[self.x_name]['{}_{}'.format(
            self.fold_prefix, self.folds[0])]
        self.y_cur = self.hf[self.y_name]['{}_{}'.format(
            self.fold_prefix, self.folds[0])]

        # Get the total length of the fold
        self.fold_len = len(self.y_cur)

    def next_fold(self):
        # Reset index
        self.index = 0

        # Remove previous fold index and move to next one
        self.folds.pop(0)

        # Cache the next fold
        self.x_cur = self.hf[self.x_name]['{}{}'.format(
            self.fold_prefix, self.folds[0])]
        self.y_cur = self.hf[self.y_name]['{}{}'.format(
            self.fold_prefix, self.folds[0])]

        # Recalculate the total length
        self.fold_len = len(self.y_cur)

    def generate(self):
        while True:
            index = self.index

            # When all batches of data are yielded, move to next fold
            if index >= self.batch_size:
                self.next_fold()

            # The last batch of data may not have less than batch_size items
            if index + self.batch_size >= self.fold_len:
                batch_x = self.x_cur[index:]
                batch_y = self.y_cur[index:]
            else:
                # Take the next batch
                batch_x = self.x_cur[index:(index + self.batch_size)]
                batch_y = self.y_cur[index:(index + self.batch_size)]

            self.index += self.batch_size
            yield batch_x, batch_y
