# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from deoxys.data.data_reader import DataReaders, DataReader
from deoxys.customize import register_datareader, \
    unregister_datareader, custom_datareader
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


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
