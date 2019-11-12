# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.callbacks import *
from ..utils import Singleton, write_file


class TestCheckpoint(Callback):  # noqa: F405
    """
    Evaluate test after every epoch
    """

    def __init__(self, model, test_history, filepath=None, period=1):
        super().__init__()
        self.deoxys_model = model
        self.test_history = test_history
        self.period = period
        self.epochs_since_last_save = 0
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            score = self.deoxys_model.evaluate_test()
            self.test_history.append(
                [epoch] + score)
            if self.filepath:
                write_file(str(self.test_history),
                           '{}.{:02d}.txt'.format(self.filepath, epoch))


class Callbacks(metaclass=Singleton):
    """
    A singleton that contains all the registered customized callbacks
    """

    def __init__(self):
        self._callbacks = {
        }

    def register(self, key, callback):
        if not issubclass(callback, Callback):  # noqa: F405
            raise ValueError(
                "The customized callback has to be a subclass"
                + " of keras.callbacks.Callback"
            )

        if key in self._callbacks:
            raise KeyError(
                "Duplicated key, please use another key for this callback"
            )
        else:
            self._callbacks[key] = callback

    def unregister(self, key):
        if key in self._callbacks:
            del self._callbacks[key]

    @property
    def callbacks(self):
        return self._callbacks


def register_callback(key, callback):
    """
    Register the customized callback.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the callback
    :type key: str
    :param callback: the customized callback class
    :type callback: keras.callbacks.Callback
    """
    Callbacks().register(key, callback)


def unregister_callback(key):
    """
    Remove the registered callback with the key-name

    :param key: the key-name of the callback to be removed
    :type key: str
    """
    Callbacks().unregister(key)


def callback_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define callback')

    if 'config' not in config:
        # auto add empty config for callback with only class_name
        config['config'] = {}
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=Callbacks().callbacks,
                                    printable_module_name='callback')
