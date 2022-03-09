# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..keras.layers import Layer
from ..keras.models import model_from_config
from ..utils import Singleton
from .activations import Activations


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

    Parameters
    ----------
    key : str
        The unique key-name of the layer
    layer : tensorflow.keras.layers.Layer
        The customized layer class
    """
    Layers().register(key, layer)


def unregister_layer(key):
    """
    Remove the registered layer with the key-name

    Parameters
    ----------
    key : str
        The key-name of the layer to be removed
    """
    Layers().unregister(key)


def layer_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define layer')

    if 'config' not in config:
        # auto add empty config for layer with only class_name
        config['config'] = {}

    if 'name' not in config['config'] and 'name' in config:
        config['config']['name'] = config['name']

    return model_from_config(
        config,
        custom_objects={**Layers().layers,
                        **Activations().activations})
