# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import deserialize
from ..utils import Singleton


class Activations(metaclass=Singleton):
    """
    A singleton that contains all the registered customized activations
    """

    def __init__(self):
        self._activations = {}

    def register(self, key, activation):
        if not issubclass(activation, Activation):
            raise ValueError(
                "The customized activation has to be a subclass"
                + " of keras.activations.Activation"
            )

        if key in self._activations:
            raise KeyError(
                "Duplicated key, please use another key for this activation"
            )
        else:
            self._activations[key] = activation

    def unregister(self, key):
        if key in self._activations:
            del self._activations[key]

    @property
    def activations(self):
        return self._activations


def register_activation(key, activation):
    """
    Register the customized activation.
    If the key name is already registered, it will raise a KeyError exception

    Parameters
    ----------
    key: str
        The unique key-name of the activation
    activation: tensorflow.keras.activations.Activation
        The customized activation class
    """
    Activations().register(key, activation)


def unregister_activation(key):
    """
    Remove the registered activation with the key-name

    Parameters
    ----------
    key: str
        The key-name of the activation to be removed
    """
    Activations().unregister(key)


def activation_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define activation')

    if 'config' not in config:
        # auto add empty config for activation with only class_name
        config['config'] = {}
    return deserialize(config, custom_objects=Activations().activations)
