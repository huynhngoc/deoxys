# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.layers import Layer
from ..utils import Singleton


class Layers(metaclass=Singleton):
    """
    A singleton that contains all the registered customized layers
    """

    def __init__(self):
        self._layers = {}

    def register(self, key, layer):
        if not issubclass(layer, Layer):
            raise ValueError(
                "The customized layer has to be a subclass"
                + " of keras.layers.Layer"
            )

        if key in self._layers:
            raise KeyError(
                "Duplicated key, please use another key name for this layer"
            )
        else:
            self._layers[key] = layer

    def unregister(self, key):
        if key in self._layers:
            del self._layers[key]

    @property
    def layers(self):
        return self._layers


def register_layer(key, layer):
    """
    Register the customized layer.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the layer
    :type key: str
    :param layer: the customized layer class
    :type layer: keras.layers.Layer
    """
    Layers().register(key, layer)


def unregister_layer(key):
    """
    Remove the registered layer with the key-name

    :param key: the key-name of the layer to be removed
    :type key: str
    """
    Layers().unregister(key)
