# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from tensorflow.keras.losses import Loss
from deoxys.model.losses import Losses
from deoxys.customize import register_loss, \
    unregister_loss, custom_loss
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def loss_class():
    class TestLoss(Loss):
        pass

    yield TestLoss


def test_is_singleton():
    losses_instance = Losses()
    another_instance = Losses()

    assert losses_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_loss("TestLoss", object)


def test_register_loss_success(loss_class):
    register_loss("TestLoss", loss_class)

    assert Losses()._losses["TestLoss"] is loss_class


def test_register_duplicate_loss(loss_class):
    register_loss("TestLoss", loss_class)

    with pytest.raises(KeyError):
        register_loss("TestLoss", loss_class)


def test_unregister_loss(loss_class):
    register_loss("TestLoss", loss_class)

    assert Losses()._losses["TestLoss"] is loss_class
    unregister_loss("TestLoss")

    assert "TestLoss" not in Losses()._losses


def test_decorator():
    @custom_loss
    class TestLoss2(Loss):
        pass

    assert Losses()._losses["TestLoss2"] is TestLoss2
