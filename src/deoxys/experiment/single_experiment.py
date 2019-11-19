# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import os
import h5py

from tensorflow.keras.callbacks import CSVLogger
from ..model.callbacks import DeoxysModelCheckpoint, EvaluationCheckpoint, \
    PredictionCheckpoint
from ..model import model_from_full_config, model_from_config, load_model
from ..utils import plot_log_performance_from_csv, mask_prediction, \
    plot_images_w_predictions


class Experiment:
    MODEL_PATH = '/model'
    MODEL_NAME = '/model.{epoch:03d}.h5'
    BEST_MODEL_PATH = '/best'
    PREDICTION_PATH = '/prediction'
    PREDICTION_NAME = '/prediction.{epoch:03d}.h5'
    LOG_FILE = '/logs.csv'
    PERFORMANCE_PATH = '/performance'
    PREDICTED_IMAGE_PATH = '/images'

    def __init__(self,
                 best_model_monitors='val_loss',
                 best_model_modes='auto',
                 log_base_path='logs'):
        self.model = None

        self.architecture = None
        self.input_params = None
        self.model_params = None
        self.train_params = None
        self.data_reader = None
        self.weights_file = None

        self.best_model_monitors = best_model_monitors
        self.best_model_modes = best_model_modes
        self.log_base_path = log_base_path

        self.best_models = {}
        self.best_saved_model = {}

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

    def run_experiment(self, train_history_log=True,
                       model_checkpoint_period=0,
                       prediction_checkpoint_period=0,
                       save_origin_images=False,
                       verbose=1,
                       epochs=None, initial_epoch=None
                       ):
        log_base_path = self.log_base_path

        if self._check_run():
            if not os.path.exists(log_base_path):
                os.makedirs(log_base_path)

            kwargs = {}

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
                        title=predicted_image_title_name,
                        contour=contour,
                        name=img_name)

        return self

    def run_lambda(self, lambda_fn, **kwargs):
        """
        Custom action between experiments

        :param lambda_fn: [description]
        :type lambda_fn: [type]
        :return: [description]
        :rtype: [type]
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
                               title='Image {index:05d}',
                               name='{index:05d}.png'):
        hf = h5py.File(data_path, 'r')
        keys = list(hf.keys())
        for index in images:
            kwargs = {key: hf[key][index] for key in keys}
            img_name = out_path + '/' + name.format(index=index, **kwargs)
            if contour:
                mask_prediction(img_name,
                                image=kwargs[base_image_name],
                                true_mask=kwargs[truth_image_name],
                                pred_mask=kwargs['predicted'],
                                title=title.format(index=index, **kwargs))
            else:
                plot_images_w_predictions(
                    img_name,
                    image=kwargs[base_image_name],
                    true_mask=kwargs[truth_image_name],
                    pred_mask=kwargs['predicted'],
                    title=title.format(index=index, **kwargs))

    def _check_run(self):
        if self.model:
            if self.model._data_reader:
                if self.model.is_compiled:
                    return True
        raise RuntimeError("Cannot run experiment with incomplete model")
        return False
