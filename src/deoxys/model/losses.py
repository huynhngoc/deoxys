# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"

from tensorflow.keras.losses import Loss, deserialize
from ..utils import Singleton


class Losses(metaclass=Singleton):
    """
    A singleton that contains all the registered customized losses
    """

    def __init__(self):
        self._losses = {}

    def register(self, key, loss):
        if not issubclass(loss, Loss):
            raise ValueError(
                "The customized loss has to be a subclass"
                + " of keras.losses.Loss"
            )

        if key in self._losses:
            raise KeyError(
                "Duplicated key, please use another key for this loss"
            )
        else:
            self._losses[key] = loss

    def unregister(self, key):
        if key in self._losses:
            del self._losses[key]

    @property
    def losses(self):
        return self._losses


def register_loss(key, loss):
    """
    Register the customized loss.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the loss
    :type key: str
    :param loss: the customized loss class
    :type loss: keras.losses.Loss
    """
    Losses().register(key, loss)


def unregister_loss(key):
    """
    Remove the registered loss with the key-name

    :param key: the key-name of the loss to be removed
    :type key: str
    """
    Losses().unregister(key)


def loss_from_config(config):
    return deserialize(config, custom_objects=Losses().losses)
