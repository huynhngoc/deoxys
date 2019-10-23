# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import model_from_config, model_from_json
from tensorflow.keras.models import load_model, save_model

from .loader import ModelLoader


class Model:
    """
    Model
    """

    def __init__(self, model, optimizer=None, loss=None, metrics=None,
                 loss_weights=None,
                 sample_weight_mode=None, weighted_metrics=None,
                 target_tensors=None, **kwargs):
        # TODO add other arguments

        self._model = model
        self._optimizer = optimizer or model.optimizer
        self._loss = loss or model.losses
        self._metrics = metrics or model.metrics
        self._loss_weights = loss_weights or model.loss_weights
        self._sample_weight_mode = sample_weight_mode or model.sample_weights
        self._weighted_metrics = weighted_metrics or model.weighted_metrics
        self._target_tensors = target_tensors or model.target_tensors

        # TODO do not compile if it's a trained model
        if (optimizer or loss or metrics or loss_weights
                or sample_weight_mode or weighted_metrics or target_tensors):
            self._model.compile(optimizer, loss, metrics, loss_weights,
                                sample_weight_mode, weighted_metrics,
                                target_tensors, **kwargs)

    @staticmethod
    def from_config(model_json, **kwargs):
        """
        Create model from JSON configuration

        :param json: [description]
        :type json: [type]
        """
        model, kwargs = ModelLoader.create(
            model_json, **kwargs).load()

        return Model(model, **kwargs)

    @staticmethod
    def from_keras_config(json, **kwarg):
        return Model(model_from_json(json), **kwarg)

    @staticmethod
    def load(filename):
        """
        Load model from file

        :param filename: path to the h5 file
        :type filename: str
        """
        return Model(load_model(filename))

    def save(self, filename):
        """
        Save model to file

        :param filename: name of the file
        :type filename: str
        """

        # TODO check if the filename str contains .h5
        self._model.save(filename)

    def fit(self, *args, **kwargs):
        """
        Train model
        """
        pass

    def predict(self):
        """
        Predict from model
        """
        pass
