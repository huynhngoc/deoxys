# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, deserialize
from ..utils import Singleton


class BinaryFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_fbeta", beta=1):
        super().__init__(reduction, name)

        self.beta = beta

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
        target_positive = tf.reduce_sum(tf.square(target), axis=reduce_ax)
        predicted_positive = tf.reduce_sum(
            tf.square(prediction), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return 1 - fb_numerator / fb_denominator


class ModifiedDiceLoss(Loss):
    def __init__(self, reduction='auto', name="modified_dice_loss", beta=1):
        super().__init__(reduction, name)

        self.beta = beta

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
        target_positive = tf.reduce_sum(target, axis=reduce_ax)
        predicted_positive = tf.reduce_sum(
            prediction, axis=reduce_ax)

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
            'BinaryFbetaLoss': BinaryFbetaLoss,
            'ModifiedDiceLoss': ModifiedDiceLoss
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

    Parameters
    ----------
    key : str
        The unique key-name of the loss
    loss : tensorflow.keras.losses.Loss
        The customized loss class
    """
    Losses().register(key, loss)


def unregister_loss(key):
    """
    Remove the registered loss with the key-name

    Parameters
    ----------
    key : str
        The key-name of the loss to be removed
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
