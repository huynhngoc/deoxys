# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"

from deoxys.keras.optimizers import Optimizer, deserialize
from ..utils import Singleton


class Optimizers(metaclass=Singleton):
    """
    A singleton that contains all the registered customized optimizers
    """

    def __init__(self):
        self._optimizers = {}

    def register(self, key, optimizer):
        if not issubclass(optimizer, Optimizer):
            raise ValueError(
                "The customized optimizer has to be a subclass"
                + " of keras.optimizers.Optimizer"
            )

        if key in self._optimizers:
            raise KeyError(
                "Duplicated key, please use another key for this optimizer"
            )
        else:
            self._optimizers[key] = optimizer

    def unregister(self, key):
        if key in self._optimizers:
            del self._optimizers[key]

    @property
    def optimizers(self):
        return self._optimizers


def register_optimizer(key, optimizer):
    """
    Register the customized optimizer.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the optimizer
    :type key: str
    :param optimizer: the customized optimizer class
    :type optimizer: keras.optimizers.Optimizer
    """
    Optimizers().register(key, optimizer)


def unregister_optimizer(key):
    """
    Remove the registered optimizer with the key-name

    :param key: the key-name of the optimizer to be removed
    :type key: str
    """
    Optimizers().unregister(key)


def optimizer_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define optimizer')

    if 'config' not in config:
        # auto add empty config for optimizer with only class_name
        config['config'] = {}
    return deserialize(config, custom_objects=Optimizers().optimizers)
