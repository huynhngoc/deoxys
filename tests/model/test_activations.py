# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.keras.layers import Activation
from deoxys.model.activations import Activations
from deoxys.customize import register_activation, \
    unregister_activation, custom_activation
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def activation_class():
    class TestActivation(Activation):
        pass

    yield TestActivation


def test_is_singleton():
    activations_instance = Activations()
    another_instance = Activations()

    assert activations_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_activation("TestActivation", object)


def test_register_activation_success(activation_class):
    register_activation("TestActivation", activation_class)

    assert Activations()._activations["TestActivation"] is activation_class


def test_register_duplicate_activation(activation_class):
    register_activation("TestActivation", activation_class)

    with pytest.raises(KeyError):
        register_activation("TestActivation", activation_class)


def test_unregister_activation(activation_class):
    register_activation("TestActivation", activation_class)

    assert Activations()._activations["TestActivation"] is activation_class
    unregister_activation("TestActivation")

    assert "TestActivation" not in Activations()._activations


def test_decorator():
    @custom_activation
    class TestActivation2(Activation):
        pass

    assert Activations()._activations["TestActivation2"] is TestActivation2
