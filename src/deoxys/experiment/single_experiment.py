# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import os
import h5py
import numpy as np
import warnings

from deoxys.keras.callbacks import CSVLogger
from ..model.callbacks import DeoxysModelCheckpoint, PredictionCheckpoint, \
    DBLogger
from ..model import model_from_full_config, model_from_config, load_model
from deoxys_vis import plot_log_performance_from_csv, mask_prediction, \
    plot_images_w_predictions, read_csv
from ..database import Tables, ExperimentAttr, HDF5Attr, SessionAttr, \
    SessionStatus


class Experiment:
    MODEL_PATH = '/model'
    MODEL_NAME = '/model.{epoch:03d}.h5'
    BEST_MODEL_PATH = '/best'
    PREDICTION_PATH = '/prediction'
    PREDICTION_NAME = '/prediction.{epoch:03d}.h5'
    LOG_FILE = '/logs.csv'
    PERFORMANCE_PATH = '/performance'
    PREDICTED_IMAGE_PATH = '/images'
    TEST_OUTPUT_PATH = '/test'
    PREDICT_TEST_NAME = '/prediction_test.h5'
    _max_size = 1

    def __init__(self,
                 log_base_path='logs',
                 best_model_monitors='val_loss',
                 best_model_modes='auto'):

        self.model = None

        self.architecture = None
        self.input_params = None
        self.model_params = None
        self.train_params = None
        self.data_reader = None
        self.weights_file = None

        self.log_base_path = log_base_path

        self.best_model_monitors = best_model_monitors if type(
            best_model_monitors) == list else [best_model_monitors]
        self.best_model_modes = best_model_modes if type(
            best_model_modes) == list else \
            [best_model_modes] * len(best_model_monitors)

        for i, (monitor, mode) in enumerate(zip(self.best_model_monitors,
                                                self.best_model_modes)):
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown, '
                              'fallback to auto mode.' % (mode),
                              RuntimeWarning)
                mode = 'auto'

            if mode == 'auto':
                if 'acc' in monitor or \
                        monitor.startswith('fmeasure') or \
                        'fbeta' in monitor:
                    self.best_model_modes[i] = 'max'
                else:
                    self.best_model_modes[i] = 'min'

    def from_full_config(self, file, weights_file=None, **kwargs):
        self.model = model_from_full_config(file,
                                            weights_file=weights_file,
                                            **kwargs)

        return self

    def from_config(self, architecture, input_params,
                    model_params, train_params,
                    dataset_params, weights_file=None):
        self.model = model_from_config(architecture, input_params,
                                       model_params, train_params,
                                       dataset_params,
                                       weights_file=weights_file)

        return self

    def from_file(self, filename):
        self.model = load_model(filename)

        return self

    def best_model(self):
        res = {}
        logger_path = self.log_base_path + self.LOG_FILE
        if os.path.isfile(logger_path):
            df = read_csv(logger_path, index_col='epoch',
                          usecols=['epoch'] + self.best_model_monitors)
            min_df = df.min()
            min_epoch = df.idxmin()
            max_df = df.max()
            max_epoch = df.idxmax()
            for monitor, mode in zip(self.best_model_monitors,
                                     self.best_model_modes):
                if mode == 'min':
                    val = min_df[monitor]
                    epoch = min_epoch[monitor]
                else:
                    val = max_df[monitor]
                    epoch = max_epoch[monitor]
                res[monitor] = {
                    'best': {
                        'val': val,
                        'epoch': epoch + 1
                    }}
        else:
            warnings.warn('No log files to check for best model')

        return res

    def run_experiment(self, train_history_log=True,
                       model_checkpoint_period=0,
                       prediction_checkpoint_period=0,
                       save_origin_images=False,
                       verbose=1,
                       epochs=None, initial_epoch=None, **custom_kwargs
                       ):
        log_base_path = self.log_base_path

        if self._check_run():
            if not os.path.exists(log_base_path):
                os.makedirs(log_base_path)

            kwargs = custom_kwargs or {}

            csv_logger_append = False

            if epochs:
                kwargs['epochs'] = epochs
            if initial_epoch:
                kwargs['initial_epoch'] = initial_epoch
                if initial_epoch > 0:
                    csv_logger_append = True

            callbacks = []

            if train_history_log:
                callback = self._create_logger(log_base_path,
                                               append=csv_logger_append)
                callbacks.append(callback)

            if model_checkpoint_period > 0:
                if not os.path.exists(log_base_path + self.MODEL_PATH):
                    os.makedirs(log_base_path + self.MODEL_PATH)

                callback = self._create_model_checkpoint(
                    log_base_path,
                    period=model_checkpoint_period)
                callbacks.append(callback)

            if prediction_checkpoint_period > 0:
                if not os.path.exists(log_base_path + self.PREDICTION_PATH):
                    os.makedirs(log_base_path + self.PREDICTION_PATH)

                callback = self._create_prediction_checkpoint(
                    log_base_path,
                    prediction_checkpoint_period,
                    use_original=save_origin_images
                )
                callbacks.append(callback)

            kwargs['callbacks'] = callbacks

            self.model.fit_train(**kwargs)

            return self

    def plot_performance(self):
        log_base_path = self.log_base_path

        if not os.path.exists(log_base_path + self.PERFORMANCE_PATH):
            os.makedirs(log_base_path + self.PERFORMANCE_PATH)

        if os.path.exists(log_base_path + self.LOG_FILE):
            print('\nPlotting performance metrics...')

            plot_log_performance_from_csv(
                filepath=log_base_path + self.LOG_FILE,
                output_path=log_base_path + self.PERFORMANCE_PATH)
        else:
            raise Warning('No log files for plotting performance')

        return self

    def plot_prediction(self, masked_images,
                        contour=True,
                        base_image_name='x',
                        truth_image_name='y',
                        predicted_image_name='predicted',
                        predicted_image_title_name='Image {index:05d}',
                        img_name='{index:05d}.png'):
        log_base_path = self.log_base_path

        if os.path.exists(log_base_path + self.PREDICTION_PATH):
            print('\nCreating prediction images...')
            # mask images
            prediced_image_path = log_base_path + self.PREDICTED_IMAGE_PATH
            if not os.path.exists(prediced_image_path):
                os.makedirs(prediced_image_path)

            for filename in os.listdir(
                    log_base_path + self.PREDICTION_PATH):
                if filename.endswith(".h5") or filename.endswith(".hdf5"):
                    # Create a folder for storing result in that period
                    images_path = prediced_image_path + '/' + filename
                    if not os.path.exists(images_path):
                        os.makedirs(images_path)

                    self._plot_predicted_images(
                        data_path=log_base_path + self.PREDICTION_PATH
                        + '/' + filename,
                        out_path=images_path,
                        images=masked_images,
                        base_image_name=base_image_name,
                        truth_image_name=truth_image_name,
                        predicted_image_name=predicted_image_name,
                        title=predicted_image_title_name,
                        contour=contour,
                        name=img_name)

        return self

    def run_test(self, use_best_model=False,
                 masked_images=None,
                 use_original_image=False,
                 contour=True,
                 base_image_name='x',
                 truth_image_name='y',
                 predicted_image_name='predicted',
                 image_name='{index:05d}.png',
                 image_title_name='Image {index:05d}'):

        log_base_path = self.log_base_path

        test_path = log_base_path + self.TEST_OUTPUT_PATH

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if use_best_model:
            raise NotImplementedError
        else:
            # score = self.model.evaluate_test(verbose=1)
            # print(score)
            filepath = test_path + self.PREDICT_TEST_NAME

            data_info = self.model.data_reader.test_generator.description
            total_size = np.product(
                data_info[0]['shape']) * data_info[0]['total'] / 1e9

            # predict directly for data of size < max_size (1GB)
            if len(data_info) == 1 and total_size < self._max_size:
                predicted = self.model.predict_test(verbose=1)
                # Create the h5 file
                hf = h5py.File(filepath, 'w')
                hf.create_dataset('predicted', data=predicted)
                hf.close()

                if use_original_image:
                    original_data = self.model.data_reader.original_test

                    for key, val in original_data.items():
                        hf = h5py.File(filepath, 'a')
                        hf.create_dataset(key, data=val)
                        hf.close()
                else:
                    # Create data from test_generator
                    x = None
                    y = None

                    test_gen = self.model.data_reader.test_generator
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

            # for large data of same size, predict each chunk
            elif len(data_info) == 1:
                test_gen = self.model.data_reader.test_generator
                data_gen = test_gen.generate()

                next_x, next_y = next(data_gen)
                predicted = self.model.predict(next_x, verbose=1)

                input_shape = (data_info[0]['total'],) + data_info[0]['shape']
                input_chunks = (1,) + data_info[0]['shape']
                target_shape = (data_info[0]['total'],) + next_y.shape[1:]
                target_chunks = (1,) + next_y.shape[1:]

                with h5py.File(filepath, 'w') as hf:
                    hf.create_dataset('x',
                                      shape=input_shape, chunks=input_chunks,
                                      compression='gzip')
                    hf.create_dataset('y',
                                      shape=target_shape, chunks=target_chunks,
                                      compression='gzip')

                    hf.create_dataset('predicted',
                                      shape=target_shape, chunks=target_chunks,
                                      compression='gzip')

                with h5py.File(filepath, 'a') as hf:
                    next_index = len(next_x)
                    hf['x'][:next_index] = next_x
                    hf['y'][:next_index] = next_y
                    hf['predicted'][:next_index] = predicted

                for _ in range(test_gen.total_batch - 1):
                    next_x, next_y = next(data_gen)
                    predicted = self.model.predict(next_x, verbose=1)

                    curr_index = next_index
                    next_index = curr_index + len(next_x)

                    with h5py.File(filepath, 'a') as hf:
                        hf['x'][curr_index:next_index] = next_x
                        hf['y'][curr_index:next_index] = next_y
                        hf['predicted'][curr_index:next_index] = predicted

            # data of different size
            else:
                test_gen = self.model.data_reader.test_generator
                data_gen = test_gen.generate()

                for curr_info_idx, info in enumerate(data_info):
                    next_x, next_y = next(data_gen)
                    predicted = self.model.predict(next_x, verbose=1)

                    input_shape = (info['total'],) + info['shape']
                    input_chunks = (1,) + info['shape']
                    target_shape = (info['total'],) + next_y.shape[1:]
                    target_chunks = (1,) + next_y.shape[1:]
                    if curr_info_idx == 0:
                        mode = 'w'
                    else:
                        mode = 'a'
                    with h5py.File(filepath, mode) as hf:
                        hf.create_dataset(f'{curr_info_idx:02d}/x',
                                          shape=input_shape,
                                          chunks=input_chunks,
                                          compression='gzip')
                        hf.create_dataset(f'{curr_info_idx:02d}/y',
                                          shape=target_shape,
                                          chunks=target_chunks,
                                          compression='gzip')

                        hf.create_dataset(f'{curr_info_idx:02d}/predicted',
                                          shape=target_shape,
                                          chunks=target_chunks,
                                          compression='gzip')

                    with h5py.File(filepath, 'a') as hf:
                        next_index = len(next_x)
                        hf[f'{curr_info_idx:02d}/x'][:next_index] = next_x
                        hf[f'{curr_info_idx:02d}/y'][:next_index] = next_y
                        hf[f'{curr_info_idx:02d}/predicted'][
                            :next_index] = predicted

                    while next_index < info['total']:
                        next_x, next_y = next(data_gen)
                        predicted = self.model.predict(next_x, verbose=1)

                        curr_index = next_index
                        next_index = curr_index + len(next_x)

                        with h5py.File(filepath, 'a') as hf:
                            hf[f'{curr_info_idx:02d}/x'][
                                curr_index:next_index] = next_x
                            hf[f'{curr_info_idx:02d}/y'][
                                curr_index:next_index] = next_y
                            hf[f'{curr_info_idx:02d}/predicted'][
                                curr_index:next_index] = predicted

            if masked_images:
                self._plot_predicted_images(
                    data_path=filepath,
                    out_path=test_path,
                    images=masked_images,
                    base_image_name=base_image_name,
                    truth_image_name=truth_image_name,
                    predicted_image_name=predicted_image_name,
                    title=image_title_name,
                    contour=contour,
                    name=image_name)

        return self

    def run_lambda(self, lambda_fn, **kwargs):
        """
        Custom action between experiments
        """
        lambda_fn(self, **kwargs)
        return self

    def _create_logger(self, base_path, append=False):
        return CSVLogger(filename=base_path + self.LOG_FILE, append=append)

    def _create_model_checkpoint(self, base_path, period):
        return DeoxysModelCheckpoint(
            period=period,
            filepath=base_path + self.MODEL_PATH + self.MODEL_NAME)

    def _create_prediction_checkpoint(self, base_path, period, use_original):
        return PredictionCheckpoint(
            filepath=base_path + self.PREDICTION_PATH + self.PREDICTION_NAME,
            period=period, use_original=use_original)

    def _plot_predicted_images(self, data_path, out_path, images,
                               contour=True,
                               base_image_name='x',
                               truth_image_name='y',
                               predicted_image_name='predicted',
                               title='Image {index:05d}',
                               name='{index:05d}.png'):
        hf = h5py.File(data_path, 'r')
        keys = list(hf.keys())
        while 'create_group' in dir(hf[keys[0]]):
            new_keys = []
            for key in keys:
                new_keys.extend([f'{key}/{k}' for k in hf[key].keys()])
            keys = new_keys

        for index in images:
            kwargs = {key: hf[key][index] for key in keys}
            img_name = out_path + '/' + name.format(index=index, **kwargs)
            if contour:
                mask_prediction(img_name,
                                image=kwargs[base_image_name],
                                true_mask=kwargs[truth_image_name],
                                pred_mask=kwargs[predicted_image_name],
                                title=title.format(index=index, **kwargs))
            else:
                plot_images_w_predictions(
                    img_name,
                    image=kwargs[base_image_name],
                    true_mask=kwargs[truth_image_name],
                    pred_mask=kwargs[predicted_image_name],
                    title=title.format(index=index, **kwargs))

    def _check_run(self):
        if self.model:
            if self.model._data_reader:
                if self.model.is_compiled:
                    return True
        raise RuntimeError("Cannot run experiment with incomplete model")
        return False


class ExperimentDB(Experiment):  # pragma: no cover

    def __init__(self, dbclient, experiment_id=None, session_id=None,
                 log_base_path='logs',
                 best_model_monitors='val_loss',
                 best_model_modes='auto'):
        """
        An experiment logging performance to a database

        Parameters
        ----------
        dbclient : deoxys.database.DBClient
            The database client
        experiment_id : str, int, or ObjectID depending of the dbclient, optional
            Experiment id, by default None
        session_id : str, int, or ObjectID depending of the dbclient, optional
            Session id, by default None
        log_base_path : str, optional
            Base path to log files, by default 'logs'
        best_model_monitors : str, optional
            Attribute to monitor, by default 'val_loss'
        best_model_modes : str, optional
            One of 'max', 'min', 'auto', by default 'auto'

        Raises
        ------
        ValueError
            When both `session_id` and `experiment_id` are not set
        """
        super().__init__(log_base_path, best_model_monitors, best_model_modes)

        self.dbclient = dbclient

        if experiment_id is None and session_id is None:
            raise ValueError('"session_id" or "experiment_id" must be set')

        if session_id:
            last_model = self.dbclient.find_max(Tables.MODELS, {
                HDF5Attr.SESSION_ID: session_id}, HDF5Attr.EPOCH)
            self.curr_epoch = last_model[HDF5Attr.EPOCH]
            self.session = dbclient.find_by_id(Tables.SESSIONS, session_id)

            self._update_epoch(last_model[HDF5Attr.EPOCH])

            self.from_file(last_model[HDF5Attr.FILE_LOCATION])
        else:
            insert_res = dbclient.insert(Tables.SESSIONS, {
                SessionAttr.EXPERIMENT_ID: experiment_id,
                SessionAttr.CURRENT_EPOCH: 0,
                SessionAttr.STATUS: 'created'
            }, time_logs=True)

            self.session = self.dbclient.find_by_id(
                Tables.SESSIONS, insert_res.inserted_id)

            self.curr_epoch = 0

            experiment_obj = self.dbclient.find_by_id(
                Tables.EXPERIMENTS, experiment_id)

            if ExperimentAttr.CONFIG in experiment_obj:
                self.from_full_config(experiment_obj[ExperimentAttr.CONFIG])
            elif ExperimentAttr.SAVED_MODEL_LOC in experiment_obj:
                self.from_file(experiment_obj[ExperimentAttr.SAVED_MODEL_LOC])

        self.log_base_path = os.path.join(
            log_base_path, str(self.dbclient.get_id(self.session)))

    def run_experiment(self, train_history_log=True,
                       model_checkpoint_period=0,
                       prediction_checkpoint_period=0,
                       save_origin_images=False,
                       verbose=1,
                       epochs=None
                       ):
        log_base_path = self.log_base_path

        if self._check_run():
            if not os.path.exists(log_base_path):
                os.makedirs(log_base_path)

            kwargs = {}

            if epochs:
                kwargs['epochs'] = epochs + self.curr_epoch

            kwargs['initial_epoch'] = self.curr_epoch

            callbacks = []

            if train_history_log:
                callback = self._create_logger(log_base_path,
                                               append=self.curr_epoch > 0)
                callbacks.append(callback)

                callback = self._create_db_logger()
                callbacks.append(callback)

            if model_checkpoint_period > 0:
                if not os.path.exists(log_base_path + self.MODEL_PATH):
                    os.makedirs(log_base_path + self.MODEL_PATH)

                callback = self._create_model_checkpoint(
                    log_base_path,
                    period=model_checkpoint_period)
                callbacks.append(callback)

            if prediction_checkpoint_period > 0:
                if not os.path.exists(log_base_path + self.PREDICTION_PATH):
                    os.makedirs(log_base_path + self.PREDICTION_PATH)

                callback = self._create_prediction_checkpoint(
                    log_base_path,
                    prediction_checkpoint_period,
                    use_original=save_origin_images
                )
                callbacks.append(callback)

            kwargs['callbacks'] = callbacks

            self._update_status(SessionStatus.TRAINING)

            try:
                self.model.fit_train(**kwargs)

                self.curr_epoch += epochs
                self._update_status(SessionStatus.FINISHED)
                self._update_epoch(self.curr_epoch)
            except Exception:
                self._update_status(SessionStatus.FAILED)

            return self

    def _create_db_logger(self):
        return DBLogger(self.dbclient, self.dbclient.get_id(self.session))

    def _update_epoch(self, new_epoch):
        self.dbclient.update_by_id(
            Tables.SESSIONS, self.dbclient.get_id(self.session),
            {SessionAttr.CURRENT_EPOCH: new_epoch}, time_logs=True)

    def _update_status(self, new_status):
        self.dbclient.update_by_id(
            Tables.SESSIONS, self.dbclient.get_id(self.session),
            {SessionAttr.STATUS: new_status}, time_logs=True)

    def _create_model_checkpoint(self, base_path, period):
        return DeoxysModelCheckpoint(
            period=period,
            filepath=base_path + self.MODEL_PATH + self.MODEL_NAME,
            dbclient=self.dbclient,
            session=self.dbclient.get_id(self.session))

    def _create_prediction_checkpoint(self, base_path, period, use_original):
        return PredictionCheckpoint(
            filepath=base_path + self.PREDICTION_PATH + self.PREDICTION_NAME,
            period=period, use_original=use_original,
            dbclient=self.dbclient,
            session=self.dbclient.get_id(self.session))
