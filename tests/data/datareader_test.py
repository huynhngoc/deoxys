# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
import h5py
import os
import numpy as np
from deoxys.data import DataReaders, DataReader, HDF5Reader, \
    datareader_from_config, BasePreprocessor
from deoxys.customize import register_datareader, \
    unregister_datareader, custom_datareader, custom_preprocessor
from deoxys.utils import Singleton


VALID_H5_FILE = 'file10.h5'
VALID_H5_FILE_M = 'file10_M.h5'
INVALID_H5_FILE_NO_GROUPS = 'file_nogroups.h5'
INVALID_H5_FILE_GROUPS = 'invalid_file.h5'


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


def setup_data():
    # Setup the hdf5 files
    data = np.arange(25000)
    data = np.reshape(data, (1000, 5, 5))
    target = [1, 2, 3, 4, 5] * 200

    return data, target


def teardown(filename=''):
    if os.path.isfile(filename):
        os.remove(filename)


@pytest.fixture(scope='module', autouse=True)
def h5file():
    # Create standardized h5 file with 10 folds, named fold_x
    filename = VALID_H5_FILE
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


@pytest.fixture(scope='module', autouse=True)
def h5file_modified():
    # Create standardized h5 file with 10 folds, named randomly
    filename = VALID_H5_FILE_M
    data, target = setup_data()

    hf = h5py.File(filename, 'w')

    for i in range(10):
        start, end = i * 100, (i + 1) * 100
        group = hf.create_group(str(i))
        group.create_dataset('input', data=data[start: end])
        group.create_dataset('target', data=target[start: end])

    hf.close()
    yield filename

    teardown(filename)


@pytest.fixture(scope='module', autouse=True)
def h5file_nogroups():
    # Create invalid h5 file with no groups
    filename = INVALID_H5_FILE_NO_GROUPS
    data, target = setup_data()

    hf = h5py.File(filename, 'w')
    hf.close()
    yield filename

    teardown(filename)


@pytest.fixture(scope='module', autouse=True)
def h5file_invalid():
    # Create invalid h5 file with different datasets in each groups
    filename = INVALID_H5_FILE_GROUPS
    data, target = setup_data()

    hf = h5py.File(filename, 'w')

    # First 3 valid groups
    for i in range(3):
        start, end = i * 100, (i + 1) * 100
        group = hf.create_group('fold_{0}'.format(i))
        group.create_dataset('input', data=data[start: end])
        group.create_dataset('target', data=target[start: end])

    # Next 3 groups with different structures
    for i in range(3, 6):
        start, end = i * 100, (i + 1) * 100
        group = hf.create_group('fold_{0}'.format(i))
        group.create_dataset('x', data=data[start: end])
        group.create_dataset('y', data=target[start: end])

    # Last 4 groups with additional data
    for i in range(6, 10):
        start, end = i * 100, (i + 1) * 100
        group = hf.create_group('fold_{0}'.format(i))
        group.create_dataset('x', data=data[start: end])
        group.create_dataset('y', data=target[start: end])
        group.create_dataset('z', data=target[start: end])

    hf.close()
    yield filename

    teardown(filename)


@pytest.fixture
def datareader_class():
    class TestDataReader(DataReader):
        pass

    yield TestDataReader


def test_is_singleton():
    datareaders_instance = DataReaders()
    another_instance = DataReaders()
    assert datareaders_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_datareader("TestDataReader", object)


def test_register_datareader_success(datareader_class):
    register_datareader("TestDataReader", datareader_class)

    assert DataReaders()._dataReaders["TestDataReader"] is datareader_class


def test_register_duplicate_datareader(datareader_class):
    register_datareader("TestDataReader", datareader_class)

    with pytest.raises(KeyError):
        register_datareader("TestDataReader", datareader_class)


def test_unregister_datareader(datareader_class):
    register_datareader("TestDataReader", datareader_class)

    assert DataReaders()._dataReaders["TestDataReader"] is datareader_class
    unregister_datareader("TestDataReader")

    assert "TestDataReader" not in DataReaders()._dataReaders


def test_decorator():
    @custom_datareader
    class TestDataReader2(DataReader):
        pass

    assert DataReaders()._dataReaders["TestDataReader2"] is TestDataReader2


def test_hdf5_dr_constructor_invalid_filename():
    with pytest.raises(ValueError):
        HDF5Reader("", batch_size=8, train_folds=None)


def test_hdf5_dr_constructor_invalid_preprocessor():
    dr = HDF5Reader(VALID_H5_FILE, batch_size=8, preprocessors=[1],
                    x_name='input', y_name='target')

    with pytest.raises(ValueError):
        dr.train_generator

    with pytest.raises(ValueError):
        dr.val_generator

    with pytest.raises(ValueError):
        dr.test_generator


def test_hdf5_dr_constructor():
    dr = HDF5Reader(VALID_H5_FILE, batch_size=8,
                    x_name='input', y_name='target', fold_prefix='fold')
    assert dr.train_folds == ['fold_0']
    assert dr.val_folds == ['fold_1']
    assert dr.test_folds == ['fold_2']
    dr.train_generator
    dr.test_generator
    dr.val_generator


def test_invalid_dr_from_config():
    with pytest.raises(ValueError):
        datareader_from_config({
            'config': {}
        })


def test_hdf5_dr_from_config():
    config = {
        'filename': VALID_H5_FILE,
        'x_name': 'input',
        'y_name': 'target',
        'train_folds': [0, 1, 2],
        'val_folds': [3],
        'test_folds': [4, 5]
    }

    dr = datareader_from_config({
        'class_name': 'HDF5Reader',
        'config': config
    })

    assert dr.train_folds == ['fold_0', 'fold_1', 'fold_2']
    assert dr.val_folds == ['fold_3']
    assert dr.test_folds == ['fold_4', 'fold_5']

    dr.train_generator
    dr.test_generator
    dr.val_generator


def test_hdf5_dr_constructor_no_prefix():
    dr = HDF5Reader(VALID_H5_FILE_M, batch_size=8,
                    x_name='input', y_name='target', fold_prefix=None)
    assert dr.train_folds == [0]
    assert dr.val_folds == [1]
    assert dr.test_folds == [2]
    dr.train_generator
    dr.test_generator
    dr.val_generator


def test_hdf5_dr_constructor_invalid_xname():
    dr = HDF5Reader(VALID_H5_FILE, batch_size=8,
                    x_name='x', y_name='target')

    with pytest.raises(RuntimeError):
        dr.train_generator


def test_hdf5_dr_constructor_invalid_yname():
    with pytest.raises(RuntimeError):
        dr = HDF5Reader(VALID_H5_FILE, batch_size=8,
                        x_name='input', y_name='y')
        dr.train_generator


def test_hdf5_dr_constructor_h5file_no_groups():
    dr = HDF5Reader(INVALID_H5_FILE_NO_GROUPS, batch_size=8,
                    x_name='input', y_name='target')
    with pytest.raises(RuntimeError):
        dr.train_generator


def test_hdf5_dr_constructor_invalid_datasetname():
    # 0 - 2: input, target
    # 3 - 5: x, y
    # 6 - 10: x, y, z
    dr = HDF5Reader(INVALID_H5_FILE_GROUPS, batch_size=8,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2], val_folds=[3, 4, 5],
                    test_folds=[6, 7, 8])

    dr.train_generator
    with pytest.raises(RuntimeError):
        dr.val_generator

    with pytest.raises(RuntimeError):
        dr.test_generator


def test_hdf5_dr_constructor_invalid_h5_structure():
    # 0 - 2: input, target
    # 3 - 5: x, y
    # 6 - 10: x, y, z
    dr = HDF5Reader(INVALID_H5_FILE_GROUPS, batch_size=8,
                    x_name='x', y_name='y',
                    train_folds=[3, 4, 5, 6], val_folds=[7],
                    test_folds=[8, 9])

    with pytest.raises(RuntimeError):
        dr.train_generator

    dr.test_generator
    dr.val_generator


def test_hdf5_dr_total_batch():
    dr = HDF5Reader(VALID_H5_FILE, batch_size=8,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4],
                    test_folds=[5, 6])

    assert dr.train_generator.total_batch == int(np.ceil(100 / 8) * 4)
    assert dr.val_generator.total_batch == int(np.ceil(100 / 8))
    assert dr.test_generator.total_batch == int(np.ceil(100 / 8) * 2)


def test_hdf5_dr_generator():
    batch_size = 8
    dr = HDF5Reader(VALID_H5_FILE, batch_size=batch_size,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4],
                    test_folds=[5, 6])

    expected_train_input = np.reshape(np.arange(400 * 25), (400, 5, 5))
    expected_train_target = np.array([1, 2, 3, 4, 5] * 80)

    total_batch = dr.train_generator.total_batch
    data_gen = dr.train_generator.generate()
    end = 0

    for i, (input_data, target) in enumerate(data_gen):
        if i >= total_batch:
            break

        start = end
        end = start + batch_size

        if end % 100 < batch_size:
            end -= end % 100

        print(start, end, expected_train_target)
        print(target)

        assert np.all(expected_train_target[start:end] == target)
        assert np.all(expected_train_input[start:end] == input_data)


def test_hdf5_dr_generator_preprocessor():
    batch_size = 8

    @custom_preprocessor
    class PlusOnePreprocessor(BasePreprocessor):
        def transform(self, x, y):
            return x + 1, y + 1

    dr = HDF5Reader(VALID_H5_FILE, batch_size=batch_size,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4],
                    test_folds=[5, 6],
                    preprocessors=PlusOnePreprocessor())

    expected_train_input = np.reshape(np.arange(1, 400 * 25 + 1), (400, 5, 5))
    expected_train_target = np.array([2, 3, 4, 5, 6] * 80)

    total_batch = dr.train_generator.total_batch
    data_gen = dr.train_generator.generate()
    end = 0

    for i, (input_data, target) in enumerate(data_gen):
        if i >= total_batch:
            break

        start = end
        end = start + batch_size

        if end % 100 < batch_size:
            end -= end % 100

        print(start, end, expected_train_target)
        print(target)

        assert np.all(expected_train_target[start:end] == target)
        assert np.all(expected_train_input[start:end] == input_data)


def test_hdf5_dr_generator_multipreprocessors():
    batch_size = 8

    @custom_preprocessor
    class PlusOnePreprocessor(BasePreprocessor):
        def transform(x, y):
            return x + 1, y + 1

    dr = HDF5Reader(VALID_H5_FILE, batch_size=batch_size,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4],
                    test_folds=[5, 6],
                    preprocessors=[PlusOnePreprocessor, PlusOnePreprocessor])

    expected_train_input = np.reshape(np.arange(2, 400 * 25 + 2), (400, 5, 5))
    expected_train_target = np.array([3, 4, 5, 6, 7] * 80)

    total_batch = dr.train_generator.total_batch
    data_gen = dr.train_generator.generate()
    end = 0

    for i, (input_data, target) in enumerate(data_gen):
        if i >= total_batch:
            break

        start = end
        end = start + batch_size

        if end % 100 < batch_size:
            end -= end % 100

        print(start, end, expected_train_target)
        print(target)

        assert np.all(expected_train_target[start:end] == target)
        assert np.all(expected_train_input[start:end] == input_data)


def test_hdf5_dr_original_test():
    batch_size = 8
    dr = HDF5Reader(VALID_H5_FILE, batch_size=batch_size,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4],
                    test_folds=[5, 6])
    original_test = dr.original_test
    assert len(original_test.keys()) == 2
    assert 'input' in original_test
    assert original_test['input'].shape[0] == 200
    assert 'target' in original_test
    assert original_test['target'].shape[0] == 200


def test_hdf5_dr_original_test_more_column():
    batch_size = 8
    dr = HDF5Reader(INVALID_H5_FILE_GROUPS, batch_size=batch_size,
                    x_name='x', y_name='y',
                    train_folds=[6, 7], val_folds=[8],
                    test_folds=[9])
    original_test = dr.original_test
    assert len(original_test.keys()) == 3
    assert 'x' in original_test
    assert original_test['x'].shape[0] == 100
    assert 'y' in original_test
    assert original_test['y'].shape[0] == 100
    assert 'z' in original_test
    assert original_test['z'].shape[0] == 100


def test_hdf5_dr_original_val():
    batch_size = 8
    dr = HDF5Reader(VALID_H5_FILE, batch_size=batch_size,
                    x_name='input', y_name='target',
                    train_folds=[0, 1, 2, 3], val_folds=[4, 5],
                    test_folds=[6, 7, 8])
    original_val = dr.original_val
    assert len(original_val.keys()) == 2
    assert 'input' in original_val
    assert original_val['input'].shape[0] == 200
    assert 'target' in original_val
    assert original_val['target'].shape[0] == 200


def test_hdf5_dr_original_val_more_column():
    batch_size = 8
    dr = HDF5Reader(INVALID_H5_FILE_GROUPS, batch_size=batch_size,
                    x_name='x', y_name='y',
                    train_folds=[6, 7], val_folds=[8],
                    test_folds=[9])
    original_val = dr.original_val
    assert len(original_val.keys()) == 3
    assert 'x' in original_val
    assert original_val['x'].shape[0] == 100
    assert 'y' in original_val
    assert original_val['y'].shape[0] == 100
    assert 'z' in original_val
    assert original_val['z'].shape[0] == 100
