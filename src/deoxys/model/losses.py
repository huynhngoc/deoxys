# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss, deserialize
import numpy as np
from ..utils import Singleton


class BinaryFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_fbeta", beta=1):
        super().__init__(reduction, name)
        self.beta = beta

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        true_positive = K.sum(prediction * target, axis=reduce_ax)
        target_positive = K.sum(K.square(target), axis=reduce_ax)
        predicted_positive = K.sum(
            K.square(prediction), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return 1 - fb_numerator / fb_denominator


class Losses(metaclass=Singleton):
    """
    A singleton that contains all the registered customized losses
    """

    def __init__(self):
        self._losses = {
            'BinaryFbetaLoss': BinaryFbetaLoss
        }

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
    if type(config) == dict:
        if 'class_name' not in config:
            raise ValueError('class_name is needed to define loss')

        if 'config' not in config:
            # auto add empty config for loss with only class_name
            config['config'] = {}
        return deserialize(
            config,
            custom_objects=Losses().losses)
    return deserialize(config, custom_objects=Losses().losses)
