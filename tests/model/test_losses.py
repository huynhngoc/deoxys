# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import Loss
from deoxys.model.losses import Losses, BinaryFbetaLoss, ModifiedDiceLoss
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


def test_binary_fbeta_loss():
    true = [
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    pred = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    true = tf.constant(true, K.floatx())
    pred = tf.constant(pred, K.floatx())
    tp = tf.constant([2, 4, 6, 5], K.floatx())
    fp = tf.constant([0, 0, 1, 1], K.floatx())
    fn = tf.constant([1, 1, 1, 0], K.floatx())

    eps = 1e-8

    def fscore(beta, tp, fp, fn):
        numerator = (1 + beta**2) * tp + eps
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp + eps

        return numerator / denominator

    beta = 1
    loss = BinaryFbetaLoss(beta=beta)
    assert np.allclose(loss.call(true, pred), (1 - fscore(beta, tp, fp, fn)))

    beta = 2
    loss = BinaryFbetaLoss(beta=beta)
    assert np.allclose(loss.call(true, pred), (1 - fscore(beta, tp, fp, fn)))


def test_dice_loss():
    true = [
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    pred = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    true = tf.constant(true, K.floatx())
    pred = tf.constant(pred, K.floatx())
    tp = tf.constant([2, 4, 6, 5], K.floatx())
    fp = tf.constant([0, 0, 1, 1], K.floatx())
    fn = tf.constant([1, 1, 1, 0], K.floatx())

    eps = 1e-8

    def fscore(beta, tp, fp, fn):
        numerator = (1 + beta**2) * tp + eps
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp + eps

        return numerator / denominator

    beta = 1
    loss = ModifiedDiceLoss(beta=beta)
    assert np.allclose(loss.call(true, pred), (1 - fscore(beta, tp, fp, fn)))

    beta = 2
    loss = ModifiedDiceLoss(beta=beta)
    assert np.allclose(loss.call(true, pred), (1 - fscore(beta, tp, fp, fn)))
