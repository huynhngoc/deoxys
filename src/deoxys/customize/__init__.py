# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..model.layers import register_layer, unregister_layer
from ..model.activations import register_activation, unregister_activation
from ..model.callbacks import register_callback, unregister_callback
from ..model.losses import register_loss, unregister_loss
from ..model.metrics import register_metric, unregister_metric
from ..model.optimizers import register_optimizer, unregister_optimizer
from ..data.data_reader import register_datareader, unregister_datareader
from ..data.preprocessor import register_preprocessor, unregister_preprocessor
from ..loaders.architecture import register_architecture


def custom_architecture(class_def):
    register_architecture(class_def.__name__, class_def)
    return class_def


def custom_layer(class_def):
    register_layer(class_def.__name__, class_def)
    return class_def


def custom_activation(class_def):
    register_activation(class_def.__name__, class_def)
    return class_def


def custom_callback(class_def):
    register_callback(class_def.__name__, class_def)
    return class_def


def custom_loss(class_def):
    register_loss(class_def.__name__, class_def)
    return class_def


def custom_metric(class_def):
    register_metric(class_def.__name__, class_def)
    return class_def


def custom_optimizer(class_def):
    register_optimizer(class_def.__name__, class_def)
    return class_def


def custom_datareader(class_def):
    register_datareader(class_def.__name__, class_def)
    return class_def


def custom_preprocessor(class_def):
    register_preprocessor(class_def.__name__, class_def)
    return class_def
