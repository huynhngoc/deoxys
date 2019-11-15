# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from tensorflow.keras.callbacks import Callback
from deoxys.model.callbacks import Callbacks
from deoxys.customize import register_callback, \
    unregister_callback, custom_callback
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def callback_class():
    class TestCallback(Callback):
        pass

    yield TestCallback


def test_is_singleton():
    callbacks_instance = Callbacks()
    another_instance = Callbacks()

    assert callbacks_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_callback("TestCallback", object)


def test_register_callback_success(callback_class):
    register_callback("TestCallback", callback_class)

    assert Callbacks()._callbacks["TestCallback"] is callback_class


def test_register_duplicate_callback(callback_class):
    register_callback("TestCallback", callback_class)

    with pytest.raises(KeyError):
        register_callback("TestCallback", callback_class)


def test_unregister_callback(callback_class):
    register_callback("TestCallback", callback_class)

    assert Callbacks()._callbacks["TestCallback"] is callback_class
    unregister_callback("TestCallback")

    assert "TestCallback" not in Callbacks()._callbacks


def test_decorator():
    @custom_callback
    class TestCallback2(Callback):
        pass

    assert Callbacks()._callbacks["TestCallback2"] is TestCallback2
