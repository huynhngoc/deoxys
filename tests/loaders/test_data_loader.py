# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
import h5py
import os
import numpy as np
from deoxys.loaders import load_data
from deoxys.customize import custom_preprocessor
from deoxys.data import BasePreprocessor, HDF5Reader
from deoxys.utils import read_file, load_json_config, Singleton


def setup_data():
    # Setup the hdf5 files
    data = np.arange(25000)
    data = np.reshape(data, (1000, 5, 5))
    target = [1, 2, 3, 4, 5] * 200

    return data, target


def teardown(filename=''):
    if os.path.isfile(filename):
        os.remove(filename)


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture(scope='module', autouse=True)
def h5file():
    # Create standardized h5 file with 10 folds, named fold_x
    filename = 'data.h5'
    data, target = setup_data()

    hf = h5py.File(filename, 'w')

    for i in range(10):
        start, end = i * 100, (i + 1) * 100
        group = hf.create_group('fold_{0}'.format(i))
        group.create_dataset('input', data=data[start: end])
        group.create_dataset('target', data=target[start: end])

    hf.close()
    yield filename
    teardown(filename)


def check_equal_data_generator(actual, expected):
    assert actual.total_batch == expected.total_batch

    total_batch = actual.total_batch

    for i, (actual_batch, expected_batch) in enumerate(
            zip(actual.generate(), expected.generate())):
        if i >= total_batch:
            break

        assert np.all(actual_batch[0] == expected_batch[0])
        assert np.all(actual_batch[1] == expected_batch[1])


def test_load_data(h5file):
    @custom_preprocessor
    class PlusOnePreprocessor(BasePreprocessor):
        def transform(self, x, y):
            return x + 1, y + 1

    actual_dr = load_data(load_json_config(
        read_file('tests/json/dataset_config.json')))

    expected_dr = HDF5Reader(
        filename=h5file,
        batch_size=8,
        preprocessors=PlusOnePreprocessor(),
        x_name='input',
        y_name='target',
        train_folds=[0, 1, 2],
        val_folds=[3],
        test_folds=[4, 5]
    )

    assert isinstance(actual_dr, HDF5Reader)
    assert isinstance(actual_dr.preprocessors[0], PlusOnePreprocessor)

    actual_train_data = actual_dr.train_generator
    actual_val_data = actual_dr.val_generator
    actual_test_data = actual_dr.test_generator

    expected_train_data = expected_dr.train_generator
    expected_val_data = expected_dr.val_generator
    expected_test_data = expected_dr.test_generator

    check_equal_data_generator(actual_train_data, expected_train_data)
    check_equal_data_generator(actual_val_data, expected_val_data)
    check_equal_data_generator(actual_test_data, expected_test_data)


def test_load_data_multi_preprocessor(h5file):
    @custom_preprocessor
    class PlusOnePreprocessor(BasePreprocessor):
        def transform(self, x, y):
            return x + 1, y + 1

    actual_dr = load_data(load_json_config(
        read_file('tests/json/dataset_config_multi_preprocessor.json')))

    expected_dr = HDF5Reader(
        filename=h5file,
        batch_size=8,
        preprocessors=[PlusOnePreprocessor(), PlusOnePreprocessor()],
        x_name='input',
        y_name='target',
        train_folds=[0, 1, 2],
        val_folds=[3],
        test_folds=[4, 5]
    )

    assert isinstance(actual_dr, HDF5Reader)
    assert isinstance(actual_dr.preprocessors[0], PlusOnePreprocessor)
    assert isinstance(actual_dr.preprocessors[1], PlusOnePreprocessor)

    actual_train_data = actual_dr.train_generator
    actual_val_data = actual_dr.val_generator
    actual_test_data = actual_dr.test_generator

    expected_train_data = expected_dr.train_generator
    expected_val_data = expected_dr.val_generator
    expected_test_data = expected_dr.test_generator

    check_equal_data_generator(actual_train_data, expected_train_data)
    check_equal_data_generator(actual_val_data, expected_val_data)
    check_equal_data_generator(actual_test_data, expected_test_data)
