# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.callbacks import *
import numpy as np
import io
import csv
import os
import h5py
from collections import OrderedDict, Iterable

from ..utils import Singleton


class DeoxysModelCallback(Callback):  # noqa: F405
    def __init__(self):
        super().__init__()
        self.deoxys_model = None

    def set_deoxys_model(self, deoxys_model):
        if not self.deoxys_model:
            self.deoxys_model = deoxys_model


class EvaluationCheckpoint(DeoxysModelCallback):
    """
    Evaluate test after every epoch
    """

    def __init__(self, filename=None, period=1,
                 separator=',', append=False):

        self.period = period
        self.epochs_since_last_save = 0

        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super().__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            score = self.deoxys_model.evaluate_test()

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, str):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    if k.ndim == 1:
                        return k[0]
                    else:
                        return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = [key for key in list(logs.keys())
                             if 'val_' not in key]

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail in this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA')
                             for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys

                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            row_dict.update(
                (key, handle_value(score[i]))
                for i, key in enumerate(self.keys) if i < len(score))
            self.writer.writerow(row_dict)
            self.csv_file.flush()


class PredictionCheckpoint(DeoxysModelCallback):
    """
    Predict test in every number of epochs
    """

    def __init__(self, filepath=None, period=1, use_original=False):
        self.period = period
        self.epochs_since_last_save = 0

        self.filepath = filepath
        self.use_original = use_original

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            # Predict all data
            predicted = self.deoxys_model.predict_test()

            # Get file name
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            # Create the h5 file
            hf = h5py.File(filepath, 'w')
            hf.create_dataset('predicted', data=predicted)
            hf.close()

            if self.use_original:
                original_data = self.deoxys_model.data_reader.original_test

                for key, val in original_data.items():
                    hf = h5py.File(filepath, 'a')
                    hf.create_dataset(key, data=val)
                    hf.close()
            else:
                # Create data from test_generator
                x = None
                y = None

                test_gen = self.deoxys_model.data_reader.test_generator
                data_gen = test_gen.generate()

                for _ in range(test_gen.total_batch):
                    next_x, next_y = next(data_gen)
                    if x is None:
                        x = next_x
                        y = next_y
                    else:
                        x = np.concatenate((x, next_x))
                        y = np.concatenate((y, next_y))

                hf = h5py.File(filepath, 'a')
                hf.create_dataset('x', data=x)
                hf.create_dataset('y', data=y)
                hf.close()


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
