# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import h5py
import numpy as np
# from tensorflow.keras.preprocessing import ImageDataGenerator
from .data_generator import DataGenerator, HDF5DataGenerator, \
    H5DataGenerator, H5PatchGenerator, H5MultiDataGenerator
from ..utils import Singleton, file_finder


class DataReader:
    """
    The base class of the Data Reader. Any newly created DataReader will
    inherit from this class.
    """

    def __init__(self, *args, **kwargs):
        # the existence of the data reader is True by default
        # if the data reader cannot be loaded because of IO reason,
        # set this value to false
        self.ready = True

    @property
    def train_generator(self):
        """
        Data Generator for the training dataset

        Returns
        -------
        deoxys.data.DataGenerator
            An DataGenerator instance that generates the train dataset
        """
        return DataGenerator().generate()

    @property
    def test_generator(self):
        """
        Data Generator for the test dataset

        Returns
        -------
        deoxys.data.DataGenerator
            An DataGenerator instance that generates the test dataset
        """
        return DataGenerator().generate()

    @property
    def val_generator(self):
        """
        Data Generator for the validation dataset

        Returns
        -------
        deoxys.data.DataGenerator
            An DataGenerator instance that generates the validation dataset
        """
        return DataGenerator().generate()

    @property
    def original_test(self):
        pass


class HDF5Reader(DataReader):
    """DataReader that use data from an hdf5 file.

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

        Parameters
        ----------
        filename : str
            The hdf5 file name that contains the data.
        batch_size : int, optional
            Number of sample to feeds in
            the neural network in each step, by default 32
        preprocessors : list of deoxys.data.Preprocessor, optional
            List of preprocessors to apply on the data, by default None
        x_name : str, optional
            Dataset name to be use as input, by default 'x'
        y_name : str, optional
            Dataset name to be use as target, by default 'y'
        batch_cache : int, optional
            Number of batches to be cached when reading the
            file, by default 10
        train_folds : list of int, or list of str, optional
            List of folds to be use as train data, by default None
        test_folds : list of int, or list of str, optional
            List of folds to be use as test data, by default None
        val_folds : list of int, or list of str, optional
            List of folds to be use as validation data, by default None
        fold_prefix : str, optional
            The prefix of the group name in the HDF5 file, by default 'fold'
    """

    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold'):
        """
        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.
        """
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5py.File(h5_filename, 'r')
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

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return HDF5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
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


class H5Reader(DataReader):
    """DataReader that use data from an hdf5 file.

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

        Parameters
        ----------
        filename : str
            The hdf5 file name that contains the data.
        batch_size : int, optional
            Number of sample to feeds in
            the neural network in each step, by default 32
        preprocessors : list of deoxys.data.Preprocessor, optional
            List of preprocessors to apply on the data, by default None
        x_name : str, optional
            Dataset name to be use as input, by default 'x'
        y_name : str, optional
            Dataset name to be use as target, by default 'y'
        batch_cache : int, optional
            Number of batches to be cached when reading the
            file, by default 10
        train_folds : list of int, or list of str, optional
            List of folds to be use as train data, by default None
        test_folds : list of int, or list of str, optional
            List of folds to be use as test data, by default None
        val_folds : list of int, or list of str, optional
            List of folds to be use as validation data, by default None
        fold_prefix : str, optional
            The prefix of the group name in the HDF5 file, by default 'fold'
        shuffle : bool, optional
            shuffle data while training, by default False
        augmentations : list of deoxys.data.Preprocessor, optional
            apply augmentation when generating traing data, by default None
    """

    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold', shuffle=False, augmentations=None):
        """
        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.
        """
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5py.File(h5_filename, 'r')
        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.shuffle = shuffle

        self.preprocessors = preprocessors
        self.augmentations = augmentations

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

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds, shuffle=self.shuffle,
            augmentations=self.augmentations)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds, shuffle=False)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5DataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds, shuffle=False)

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


class H5PatchReader(DataReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold',
                 patch_size=128, overlap=0.5, shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0.1, check_drop_channel=None,
                 bounding_box=False):
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5_filename

        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.shuffle = shuffle

        self.patch_size = patch_size
        self.overlap = overlap

        self.preprocess_first = preprocess_first
        self.drop_fraction = drop_fraction
        self.check_drop_channel = check_drop_channel
        self.bounding_box = bounding_box

        self.preprocessors = preprocessors
        self.augmentations = augmentations

        if preprocessors:
            if '__iter__' not in dir(preprocessors):
                self.preprocessors = [preprocessors]

        if augmentations:
            if '__iter__' not in dir(augmentations):
                self.augmentations = [augmentations]

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

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=self.shuffle,
            augmentations=self.augmentations,
            preprocess_first=self.preprocess_first,
            drop_fraction=self.drop_fraction,
            check_drop_channel=self.check_drop_channel,
            bounding_box=self.bounding_box)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5PatchGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)

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


class H5MultiReader(DataReader):  # pragma: no cover
    """DataReader that use data from an hdf5 file.

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

        Parameters
        ----------
        filename : str
            The hdf5 file name that contains the data.
        batch_size : int, optional
            Number of sample to feeds in
            the neural network in each step, by default 32
        preprocessors : list of deoxys.data.Preprocessor, optional
            List of preprocessors to apply on the data, by default None
        x_name : str, optional
            Dataset name to be use as input, by default 'x'
        y_name : str, optional
            Dataset name to be use as target, by default 'y'
        batch_cache : int, optional
            Number of batches to be cached when reading the
            file, by default 10
        train_folds : list of int, or list of str, optional
            List of folds to be use as train data, by default None
        test_folds : list of int, or list of str, optional
            List of folds to be use as test data, by default None
        val_folds : list of int, or list of str, optional
            List of folds to be use as validation data, by default None
        fold_prefix : str, optional
            The prefix of the group name in the HDF5 file, by default 'fold'
        shuffle : bool, optional
            shuffle data while training, by default False
        augmentations : list of deoxys.data.Preprocessor, optional
            apply augmentation when generating traing data, by default None
    """

    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold', shuffle=False, augmentations=None,
                 other_input_names=None, other_preprocessors=None,
                 other_augmentations=None):
        """
        Initialize a HDF5 Data Reader, which reads data from a HDF5
        file. This file should be split into groups. Each group contain
        datasets, each of which is a column in the data.
        """
        super().__init__()

        h5_filename = file_finder(filename)
        if h5_filename is None:
            # HDF5DataReader is created, but won't be loaded into model
            self.ready = False
            return

        self.hf = h5py.File(h5_filename, 'r')
        self.batch_size = batch_size
        self.batch_cache = batch_cache

        self.shuffle = shuffle

        self.preprocessors = preprocessors
        self.augmentations = augmentations

        self.x_name = x_name
        self.y_name = y_name
        self.fold_prefix = fold_prefix

        self.other_input_names = other_input_names
        self.other_preprocessors = other_preprocessors
        self.other_augmentations = other_augmentations

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

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds, shuffle=self.shuffle,
            augmentations=self.augmentations,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors,
            other_augmentations=self.other_augmentations)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds, shuffle=False,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5MultiDataGenerator(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds, shuffle=False,
            other_input_names=self.other_input_names,
            other_preprocessors=self.other_preprocessors)

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
            'HDF5Reader': HDF5Reader,
            'H5Reader': H5Reader,
            'H5PatchReader': H5PatchReader,
            'H5MultiInputReader': H5MultiReader
        }

    def register(self, key, dr):
        if not issubclass(dr, DataReader):
            raise ValueError(
                "The customized data reader has to be a subclass"
                + " of deoxys.data.DataReader"
            )

        if key in self._dataReaders:
            raise KeyError(
                "Duplicated key, please use another key for this data reader"
            )
        else:
            self._dataReaders[key] = dr

    def unregister(self, key):
        if key in self._dataReaders:
            del self._dataReaders[key]

    @property
    def data_readers(self):
        return self._dataReaders


def register_datareader(key, dr):
    """Register the customized data reader.
    If the key name is already registered, it will raise a KeyError exception.

    Parameters
    ----------
    key : str
        The unique key-name of the data reader
    dr : deoxys.data.DataReader
        The customized data reader class
    """
    DataReaders().register(key, dr)


def unregister_datareader(key):
    """
    Remove the registered data reader with the key-name

    Parameters
    ----------
    key : str
        The key-name of the data reader to be removed
    """
    DataReaders().unregister(key)


def _deserialize(config, custom_objects={}):
    return custom_objects[config['class_name']](**config['config'])


def datareader_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define data reader')

    if 'config' not in config:
        # auto add empty config for data reader with only class_name
        config['config'] = {}
    return _deserialize(config, custom_objects=DataReaders().data_readers)
