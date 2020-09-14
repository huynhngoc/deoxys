# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.keras.optimizers import Optimizer
from deoxys.model.optimizers import Optimizers
from deoxys.customize import register_optimizer, \
    unregister_optimizer, custom_optimizer
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def optimizer_class():
    class TestOptimizer(Optimizer):
        pass

    yield TestOptimizer


def test_is_singleton():
    optimizers_instance = Optimizers()
    another_instance = Optimizers()

    assert optimizers_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_optimizer("TestOptimizer", object)


def test_register_optimizer_success(optimizer_class):
    register_optimizer("TestOptimizer", optimizer_class)

    assert Optimizers()._optimizers["TestOptimizer"] is optimizer_class


def test_register_duplicate_optimizer(optimizer_class):
    register_optimizer("TestOptimizer", optimizer_class)

    with pytest.raises(KeyError):
        register_optimizer("TestOptimizer", optimizer_class)


def test_unregister_optimizer(optimizer_class):
    register_optimizer("TestOptimizer", optimizer_class)

    assert Optimizers()._optimizers["TestOptimizer"] is optimizer_class
    unregister_optimizer("TestOptimizer")

    assert "TestOptimizer" not in Optimizers()._optimizers


def test_decorator():
    @custom_optimizer
    class TestOptimizer2(Optimizer):
        pass

    assert Optimizers()._optimizers["TestOptimizer2"] is TestOptimizer2
