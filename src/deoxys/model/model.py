# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import Model as KerasModel, \
    model_from_config as keras_model_from_config, \
    model_from_json as keras_model_from_json, \
    load_model as keras_load_model, \
    save_model as keras_save_model

from ..loaders import load_architecture, load_params
from ..utils import load_json_config


class Model:
    """
    Model
    """

    def __init__(self, model, model_params=None, train_params=None,
                 pre_compiled=False):
        # TODO add other arguments

        self._model = model
        self._model_params = model_params
        self._train_parms = train_params
        self._compiled = pre_compiled

        if model_params:
            if 'optimizer' in model_params:
                self._model.compile(**model_params)
                self._compiled = True
            else:
                raise ValueError('optimizer is a required parameter.')

    def compile(self, optimizer=None, loss=None, metrics=None,
                loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None,
                target_tensors=None, **kwargs):
        if self._compiled:
            raise Warning(
                'This will override the previous configuration of the model.')

        self._model.compile(optimizer, loss, metrics,
                            loss_weights,
                            sample_weight_mode, weighted_metrics,
                            target_tensors, **kwargs)
        self._compiled = True

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
        self._model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict from model
        """
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._model.evaluate(*args, **kwargs)

    @property
    def is_compiled(self):
        return self._compiled


def model_from_full_config(model_config, **kwargs):
    """
    Return the model from the full config

    :param model_config: a json string or a dictionary contains the
    architecture, model_params, input_params of the model
    :type model_config: a JSON string or a dictionary object
    :return: The model
    :rtype: deoxys.model.Model
    """
    config, = load_json_config(model_config)

    if ('architecture' not in config or 'input_params' not in config):
        raise ValueError('architecture and input_params are required')

    return model_from_config(
        config['architecture'],
        config['input_params'],
        config['model_params'] if 'model_params' in config else None,
        config['train_params'] if 'train_params' in config else None,
        **kwargs)


def model_from_config(architecture, input_params,
                      model_params=None, train_params=None,
                      **kwargs):
    architecture, input_params, model_params = load_json_config(
        architecture, input_params, model_params)

    # load the model based on the architecture type (Unet / Dense/ Sequential)
    loaded_model = load_architecture(architecture, input_params)

    # Load the parameters to compile the model
    loaded_params = load_params(model_params)

    # the keyword arguments will replace existing params
    loaded_params.update(kwargs)

    return Model(loaded_model, loaded_params, train_params, **kwargs)


def model_from_keras_config(config, **kwarg):
    return Model(keras_model_from_config(config), **kwarg)


def model_from_keras_json(json, **kwarg):
    return Model(keras_model_from_json(json), **kwarg)


def load_model(filename):
    """
    Load model from file

    :param filename: path to the h5 file
    :type filename: str
    """
    return Model(keras_load_model(filename), pre_compiled=True)
