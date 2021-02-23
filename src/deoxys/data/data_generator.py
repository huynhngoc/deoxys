# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import numpy as np


class DataGenerator:
    def __init__(self):
        pass

    @property
    def total_batch(self):
        """
        Total number of batches to iterate all data

        Returns
        -------
        int
            Total number of batches to iterate all data
        """
        return 0

    def generate(self):
        raise NotImplementedError()

    @property
    def description(self):
        """
        Description of the size and number of input items in the data

        Returns
        -------
        list of dictionary
            List of information
            ```
            [{
                'shape': (128, 128, 2),
                'total': 100
            },{
                'shape': (256, 256, 2),
                'total': 100
            }]
            ```
        """
        return None


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

        if self.preprocessors:
            if type(self.preprocessors) == list:
                for preprocessor in self.preprocessors:
                    self.x_cur, self.y_cur = preprocessor.transform(
                        self.x_cur, self.y_cur)
            else:
                self.x_cur, self.y_cur = self.preprocessors.transform(
                    self.x_cur, self.y_cur)

        # Get the total length of the first fold
        self.fold_len = len(self.hf[first_fold_name][self.y_name])

        self._total_batch = None
        self._description = None

    @property
    def description(self):
        if self._description is None:
            fold_names = self.folds
            description = []
            # find the shape of the inputs in the first fold
            shape = self.hf[fold_names[0]][self.x_name].shape
            obj = {'shape': shape[1:], 'total': shape[0]}

            for fold_name in fold_names[1:]:  # iterate through each fold
                shape = self.hf[fold_name][self.x_name].shape
                # if the shape are the same, increase the total number
                if np.all(obj['shape'] == shape[1:]):
                    obj['total'] += shape[0]
                # else create a new item
                else:
                    description.append(obj.copy())
                    obj = {'shape': shape[1:], 'total': shape[0]}

            # append the last item
            description.append(obj.copy())

            self._description = description
        return self._description

    @property
    def total_batch(self):
        """Total number of batches to iterate all data.
        It will be used as the number of steps per epochs when training or
        validating data in a model.

        Returns
        -------
        int
            Total number of batches to iterate all data
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

        # Apply preprocessor
        if self.preprocessors:
            if type(self.preprocessors) == list:
                for preprocessor in self.preprocessors:
                    self.x_cur, self.y_cur = preprocessor.transform(
                        self.x_cur, self.y_cur)
            else:
                self.x_cur, self.y_cur = self.preprocessors.transform(
                    self.x_cur, self.y_cur)

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
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

            self.index += self.batch_size
            yield batch_x, batch_y


class H5DataGenerator(DataGenerator):
    def __init__(self, h5file, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y', folds=None,
                 shuffle=False, augmentations=None):
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

        # Checking for valid preprocessor
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

        if augmentations:
            if type(augmentations) == list:
                for pp in augmentations:
                    if not callable(getattr(pp, 'transform', None)):
                        raise ValueError(
                            'Augmentation must be a preprocessor with'
                            ' a "transform" method')
            else:
                if not callable(getattr(augmentations, 'transform', None)):
                    raise ValueError(
                        'Augmentation must be a preprocessor with'
                        ' a "transform" method')

        self.hf = h5file
        self.batch_size = batch_size
        self.seg_size = batch_size * batch_cache
        self.preprocessors = preprocessors
        self.augmentations = augmentations

        self.x_name = x_name
        self.y_name = y_name

        self.shuffle = shuffle

        self.folds = str_folds

        self._total_batch = None
        self._description = None

        # initialize "index" of current seg and fold
        self.seg_idx = 0
        self.fold_idx = 0

        # shuffle the folds
        if self.shuffle:
            np.random.shuffle(self.folds)

        # calculate number of segs in this fold
        seg_num = np.ceil(
            h5file[self.folds[0]][y_name].shape[0] / self.seg_size)

        self.seg_list = np.arange(seg_num).astype(int)
        if self.shuffle:
            np.random.shuffle(self.seg_list)

    @property
    def description(self):
        if self.shuffle:
            raise Warning('The data is shuffled, the description results '
                          'may not accurate')
        if self._description is None:
            fold_names = self.folds
            description = []
            # find the shape of the inputs in the first fold
            shape = self.hf[fold_names[0]][self.x_name].shape
            obj = {'shape': shape[1:], 'total': shape[0]}

            for fold_name in fold_names[1:]:  # iterate through each fold
                shape = self.hf[fold_name][self.x_name].shape
                # if the shape are the same, increase the total number
                if np.all(obj['shape'] == shape[1:]):
                    obj['total'] += shape[0]
                # else create a new item
                else:
                    description.append(obj.copy())
                    obj = {'shape': shape[1:], 'total': shape[0]}

            # append the last item
            description.append(obj.copy())

            self._description = description
        return self._description

    @property
    def total_batch(self):
        """Total number of batches to iterate all data.
        It will be used as the number of steps per epochs when training or
        validating data in a model.

        Returns
        -------
        int
            Total number of batches to iterate all data
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
        self.fold_idx += 1

        if self.fold_idx == len(self.folds):
            self.fold_idx = 0

            if self.shuffle:
                np.random.shuffle(self.folds)

    def next_seg(self):
        if self.seg_idx == len(self.seg_list):
            # move to next fold
            self.next_fold()

            # reset seg index
            self.seg_idx = 0
            # recalculate seg_num
            cur_fold = self.folds[self.fold_idx]
            seg_num = np.ceil(
                self.hf[cur_fold][self.y_name].shape[0] / self.seg_size)

            self.seg_list = np.arange(seg_num).astype(int)

            if self.shuffle:
                np.random.shuffle(self.seg_list)

        cur_fold = self.folds[self.fold_idx]
        cur_seg_idx = self.seg_list[self.seg_idx]

        start, end = cur_seg_idx * \
            self.seg_size, (cur_seg_idx + 1) * self.seg_size

        # print(cur_fold, cur_seg_idx, start, end)

        seg_x = self.hf[cur_fold][self.x_name][start: end]
        seg_y = self.hf[cur_fold][self.y_name][start: end]

        return_indice = np.arange(len(seg_y))

        if self.shuffle:
            np.random.shuffle(return_indice)

        # Apply preprocessor
        if self.preprocessors:
            if type(self.preprocessors) == list:
                for preprocessor in self.preprocessors:
                    seg_x, seg_y = preprocessor.transform(
                        seg_x, seg_y)
            else:
                seg_x, seg_y = self.preprocessors.transform(
                    seg_x, seg_y)
        # Apply augmentation:
        if self.augmentations:
            if type(self.augmentations) == list:
                for preprocessor in self.augmentations:
                    seg_x, seg_y = preprocessor.transform(
                        seg_x, seg_y)
            else:
                seg_x, seg_y = self.augmentations.transform(
                    seg_x, seg_y)

        # increase seg index
        self.seg_idx += 1

        return seg_x[return_indice], seg_y[return_indice]

    def generate(self):
        """Create a generator that generate a batch of data

        Yields
        -------
        tuple of 2 arrays
            batch of (input, target)
        """
        while True:
            seg_x, seg_y = self.next_seg()

            seg_len = len(seg_y)

            for i in range(0, seg_len, self.batch_size):
                batch_x = seg_x[i:(i + self.batch_size)]
                batch_y = seg_y[i:(i + self.batch_size)]

                # print(batch_x.shape)

                yield batch_x, batch_y
