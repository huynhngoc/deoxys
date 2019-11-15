# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from deoxys.data import Preprocessors, BasePreprocessor
from deoxys.customize import register_preprocessor, \
    unregister_preprocessor, custom_preprocessor
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def preprocessor_class():
    class TestPreprocessor(BasePreprocessor):
        pass

    yield TestPreprocessor


def test_is_singleton():
    preprocessors_instance = Preprocessors()
    another_instance = Preprocessors()

    assert preprocessors_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_preprocessor("TestPreprocessor", object)


def test_register_preprocessor_success(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    assert Preprocessors(
    )._preprocessors["TestPreprocessor"] is preprocessor_class


def test_register_duplicate_preprocessor(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    with pytest.raises(KeyError):
        register_preprocessor("TestPreprocessor", preprocessor_class)


def test_unregister_preprocessor(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    assert Preprocessors(
    )._preprocessors["TestPreprocessor"] is preprocessor_class
    unregister_preprocessor("TestPreprocessor")

    assert "TestPreprocessor" not in Preprocessors()._preprocessors


def test_decorator():
    @custom_preprocessor
    class TestPreprocessor2(BasePreprocessor):
        pass

    assert Preprocessors(
    )._preprocessors["TestPreprocessor2"] is TestPreprocessor2
