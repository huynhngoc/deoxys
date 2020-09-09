# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..keras.utils import deserialize_keras_object
from ..keras.callbacks import *

import warnings
import numpy as np
import io
import csv
import os
import h5py
from collections import OrderedDict, Iterable

from ..utils import Singleton
from ..database import Tables, HDF5Attr, LogAttr


class DeoxysModelCallback(Callback):  # noqa: F405
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deoxys_model = None

    def set_deoxys_model(self, deoxys_model):
        if not self.deoxys_model:
            self.deoxys_model = deoxys_model


class EvaluationCheckpoint(DeoxysModelCallback):
    """
    Evaluate test after some epochs. Only use when cross validation
    to avoid data leakage.
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

            print('\nEvaluating test set...')
            self.epochs_since_last_save = 0
            score = self.deoxys_model.evaluate_test(verbose=1)

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


class DBLogger(Callback):  # noqa: F405

    def __init__(self, dbclient, session):
        """
        Log performance to database

        :param dbclient: the database client that stores all data
        :type dbclient: deoxys.database.DBClient
        :param session: session id
        :type session: str, int, or ObjectId, depending of the provider
                       of DBClient
        """
        self.dbclient = dbclient
        self.session = session

        self.keys = None

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                if k.ndim == 1:
                    if isinstance(k[0], np.generic):
                        return np.asscalar(k[0])
                    else:
                        return k[0]
                else:
                    return '"[%s]"' % (', '.join(map(str, k)))
            else:
                if isinstance(k, np.generic):
                    return np.asscalar(k)
                else:
                    return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that it won't fail in this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA')
                         for k in self.keys])

        identifier = {LogAttr.SESSION_ID: self.session,
                      LogAttr.EPOCH: epoch + 1}
        perf_log = OrderedDict(identifier)
        perf_log.update((key, handle_value(logs[key])) for key in self.keys)

        self.dbclient.update_insert(Tables.LOGS, identifier, perf_log)


class PredictionCheckpoint(DeoxysModelCallback):
    """
    Predict test in every number of epochs
    """

    def __init__(self, filepath=None, period=1, use_original=False,
                 dbclient=None, session=None):
        self.period = period
        self.epochs_since_last_save = 0

        self.filepath = filepath
        self.use_original = use_original

        self.dbclient = dbclient
        self.session = session

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            print('\nPredicting validation data...')

            # Predict all data
            predicted = self.deoxys_model.predict_val(verbose=1)

            # Get file name
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            # Create the h5 file
            hf = h5py.File(filepath, 'w')
            hf.create_dataset('predicted', data=predicted, compression="gzip")
            hf.close()

            if self.use_original:
                original_data = self.deoxys_model.data_reader.original_val

                for key, val in original_data.items():
                    hf = h5py.File(filepath, 'a')
                    hf.create_dataset(key, data=val, compression="gzip")
                    hf.close()
            else:
                # Create data from val_generator
                x = None
                y = None

                val_gen = self.deoxys_model.data_reader.val_generator
                data_gen = val_gen.generate()

                for _ in range(val_gen.total_batch):
                    next_x, next_y = next(data_gen)
                    if x is None:
                        x = next_x
                        y = next_y
                    else:
                        x = np.concatenate((x, next_x))
                        y = np.concatenate((y, next_y))

                hf = h5py.File(filepath, 'a')
                hf.create_dataset('x', data=x, compression="gzip")
                hf.create_dataset('y', data=y, compression="gzip")
                hf.close()

            if self.dbclient:
                item = OrderedDict(
                    {HDF5Attr.SESSION_ID: self.session,
                     HDF5Attr.EPOCH: epoch + 1})
                item.update(
                    {HDF5Attr.FILE_LOCATION: os.path.abspath(filepath)})

                self.dbclient.insert(Tables.PREDICTIONS, item)


class DeoxysModelCheckpoint(DeoxysModelCallback,
                            ModelCheckpoint):  # noqa: F405

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1,
                 dbclient=None, session=None):
        super().__init__(filepath=filepath,
                         monitor=monitor, verbose=verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=mode, period=period)

        self.dbclient = dbclient
        self.session = session

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            abs_path = os.path.abspath(filepath)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model '
                                  ' only with % s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from '
                                  '%0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.deoxys_model.save(filepath, overwrite=True)

                        if self.dbclient:
                            item = OrderedDict(
                                {HDF5Attr.SESSION_ID: self.session,
                                 HDF5Attr.EPOCH: epoch + 1})
                            item.update(
                                {HDF5Attr.FILE_LOCATION: abs_path})

                            self.dbclient.insert(Tables.MODELS, item)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from '
                                  '%0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.deoxys_model.save(filepath, overwrite=True)

                if self.dbclient:
                    item = OrderedDict(
                        {HDF5Attr.SESSION_ID: self.session,
                         HDF5Attr.EPOCH: epoch + 1})
                    item.update({HDF5Attr.FILE_LOCATION: abs_path})
                    self.dbclient.insert(Tables.MODELS, item)


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
