# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from tensorflow.keras.layers import Layer as KerasLayer
from deoxys.model.layers import Layers
from deoxys.customize import register_layer, unregister_layer, custom_layer
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def layer_class():
    class TestLayer(KerasLayer):
        pass

    yield TestLayer


def test_is_singleton():
    layers_instance = Layers()
    another_instance = Layers()

    assert layers_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_layer("TestLayer", object)


def test_register_layer_success(layer_class):
    register_layer("TestLayer", layer_class)

    assert Layers()._layers["TestLayer"] is layer_class


def test_register_duplicate_layer(layer_class):
    register_layer("TestLayer", layer_class)

    with pytest.raises(KeyError):
        register_layer("TestLayer", layer_class)


def test_unregister_layer(layer_class):
    register_layer("TestLayer", layer_class)

    assert Layers()._layers["TestLayer"] is layer_class
    unregister_layer("TestLayer")

    assert "TestLayer" not in Layers()._layers


def test_decorator():
    @custom_layer
    class TestLayer2(KerasLayer):
        pass

    assert Layers()._layers["TestLayer2"] is TestLayer2
