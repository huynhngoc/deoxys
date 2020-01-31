# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import \
    model_from_config as keras_model_from_config, \
    model_from_json as keras_model_from_json, \
    load_model as keras_load_model, Model as KerasModel

import json
import h5py

from ..loaders import load_architecture, load_params, \
    load_data, load_train_params
from ..utils import load_json_config
from .layers import Layers
from .metrics import Metrics
from .losses import Losses
from .activations import Activations
from .callbacks import DeoxysModelCallback


class Model:
    """
    Model
    """
    _evaluate_param_keys = ['callbacks', 'max_queue_size',
                            'workers', 'use_multiprocessing', 'verbose']

    _predict_param_keys = ['callbacks', 'max_queue_size',
                           'workers', 'use_multiprocessing', 'verbose']

    _fit_param_keys = ['epochs', 'verbose', 'callbacks', 'class_weight',
                       'max_queue_size', 'workers',
                       'use_multiprocessing', 'shuffle', 'initial_epoch']

    def __init__(self, model, model_params=None, train_params=None,
                 data_reader=None,
                 pre_compiled=False, weights_file=None, config=None):

        self._model = model
        self._model_params = model_params
        self._train_params = train_params
        self._compiled = pre_compiled
        self._data_reader = data_reader
        self._layers = None

        self.config = config or {}

        if model_params:
            if 'optimizer' in model_params:
                self._model.compile(**model_params)
                self._compiled = True
                if weights_file:
                    self._model.load_weights(weights_file)
            else:
                raise ValueError('optimizer is a required parameter in '
                                 'model_params.')

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

    def save(self, filename, *args, **kwargs):
        """
        Save model to file

        :param filename: name of the file
        :type filename: str
        """
        self._model.save(filename, *args, **kwargs)

        if self.config:
            config = json.dumps(self.config)

            saved_model = h5py.File(filename, 'a')
            saved_model.attrs.create('deoxys_config', config)
            saved_model.close()

    def fit(self, *args, **kwargs):
        """
        Train model
        """
        # Reset layer map as weight will change after training
        self._layers = None

        self._model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict from model
        """
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._model.evaluate(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        # Reset layer map as weight will change after training
        self._layers = None
        return self._model.fit_generator(*args, **kwargs)

    def evaluate_generator(self, *args, **kwargs):
        return self._model.evaluate_generator(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self._model.predict_generator(*args, **kwargs)

    def fit_train(self, **kwargs):
        # Reset layer map as weight will change after training
        self._layers = None

        params = self._get_train_params(self._fit_param_keys, **kwargs)
        train_data_gen = self._data_reader.train_generator
        train_steps_per_epoch = train_data_gen.total_batch

        val_data_gen = self._data_reader.val_generator
        val_steps_per_epoch = val_data_gen.total_batch

        return self.fit_generator(train_data_gen.generate(),
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_data_gen.generate(),
                                  validation_steps=val_steps_per_epoch,
                                  **params)

    def evaluate_train(self, **kwargs):
        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.train_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate_generator(data_gen.generate(),
                                       steps=steps_per_epoch,
                                       **params)

    def evaluate_val(self, **kwargs):
        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.val_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate_generator(data_gen.generate(),
                                       steps=steps_per_epoch,
                                       **params)

    def predict_val(self, **kwargs):
        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.val_generator
        steps_per_epoch = data_gen.total_batch

        return self.predict_generator(data_gen.generate(),
                                      steps=steps_per_epoch,
                                      **params)

    def evaluate_test(self, **kwargs):
        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.test_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate_generator(data_gen.generate(),
                                       steps=steps_per_epoch,
                                       **params)

    def predict_test(self, **kwargs):
        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.test_generator
        steps_per_epoch = data_gen.total_batch

        return self.predict_generator(data_gen.generate(),
                                      steps=steps_per_epoch,
                                      **params)

    @property
    def is_compiled(self):
        return self._compiled

    @property
    def model(self):
        return self._model

    @property
    def data_reader(self):
        return self._data_reader

    @property
    def layers(self):
        if self._layers is None:
            self._layers = {layer.name: layer for layer in self.model.layers}

        return self._layers

    @property
    def node_graph(self):
        """
        Node graph from nodes in model, ignoring resize and concatenate nodes
        """
        layers = self.layers

        connection = []

        def previous_layers(name):
            # Keep looking for the previous layer of resize layers
            if 'resize' in name:
                prevs = layers[name].inbound_nodes[0].get_config()[
                    'inbound_layers']
                if type(prevs) == str:
                    prev = prevs
                else:
                    # in case there are multiple layers, take the 1st one
                    prev = prevs[0]
                return previous_layers(prev)
            return name

        model = self.model

        for layer in model.layers:
            if 'resize' in layer.name or 'concatenate' in layer.name:
                continue
            inbound_nodes = layer.inbound_nodes
            for node in inbound_nodes:
                inbound_layers = node.get_config()['inbound_layers']

                if type(inbound_layers) == str:
                    if 'concatenate' in inbound_layers:
                        concat_layer = layers[inbound_layers]

                        nodes = concat_layer.inbound_nodes[0].get_config()[
                            'inbound_layers']
                        for n in nodes:
                            connection.append({
                                'from': previous_layers(n),
                                'to': layer.name})
                    else:
                        connection.append({
                            'from': layers[inbound_layers].name,
                            'to': layer.name
                        })
        return connection

    def activation_map(self, layer_name):
        return KerasModel(inputs=self.model.inputs,
                          outputs=self.layers['layer_name'].output)

    def activation_map_for_image(self, layer_name, images):
        return self.activation_map(layer_name).predict(images)

    def _get_train_params(self, keys, **kwargs):
        params = {}

        # Load train_params every run
        train_params = load_train_params(self._train_params)

        if 'callbacks' in train_params and 'callbacks' in kwargs:
            kwargs['callbacks'] = train_params['callbacks'] + \
                kwargs['callbacks'] if type(
                kwargs['callbacks']) == list else [kwargs['callbacks']]

        params.update(train_params)
        params.update(kwargs)

        if 'callbacks' in params:
            # set deoxys model for custom model
            for callback in params['callbacks']:
                if isinstance(callback, DeoxysModelCallback):
                    callback.set_deoxys_model(self)

        params = {key: params[key] for key in params if key in keys}
        return params


def model_from_full_config(model_config, **kwargs):
    """
    Return the model from the full config

    :param model_config: a json string or a dictionary contains the
    architecture, model_params, input_params of the model
    :type model_config: a JSON string or a dictionary object
    :return: The model
    :rtype: deoxys.model.Model
    """
    config = load_json_config(model_config)

    if ('architecture' not in config or 'input_params' not in config):
        raise ValueError('architecture and input_params are required')

    return model_from_config(
        config['architecture'],
        config['input_params'],
        config['model_params'] if 'model_params' in config else None,
        config['train_params'] if 'train_params' in config else None,
        config['dataset_params'] if 'dataset_params' in config else None,
        **kwargs)


def model_from_config(architecture, input_params,
                      model_params=None, train_params=None,
                      dataset_params=None, **kwargs):
    architecture, input_params, model_params, train_params = load_json_config(
        architecture, input_params, model_params, train_params)

    config = {
        'architecture': architecture,
        'input_params': input_params,
        'model_params': model_params,
        'train_params': train_params,
        'dataset_params': dataset_params
    }

    # load the model based on the architecture type (Unet / Dense/ Sequential)
    loaded_model = load_architecture(architecture, input_params)

    # Load the parameters to compile the model
    loaded_params = load_params(model_params)

    # the keyword arguments will replace existing params
    loaded_params.update(kwargs)

    # load the data generator
    data_generator = None
    if dataset_params:
        data_generator = load_data(dataset_params)

    return Model(loaded_model, loaded_params, train_params,
                 data_generator, config=config, **kwargs)


def model_from_keras_config(config, **kwarg):
    return Model(keras_model_from_config(config), **kwarg)


def model_from_keras_json(json, **kwarg):
    return Model(keras_model_from_json(json), **kwarg)


def load_model(filename, **kwargs):
    """
    Load model from file

    :param filename: path to the h5 file
    :type filename: str
    """
    # Keras got the error of loading custom object
    try:
        model = Model(keras_load_model(filename,
                                       custom_objects={
                                           **Layers().layers,
                                           **Activations().activations,
                                           **Losses().losses,
                                           **Metrics().metrics}),
                      pre_compiled=True, **kwargs)
    except Exception:
        hf = h5py.File(filename, 'r')
        if 'deoxys_config' in hf.attrs.keys():
            config = hf.attrs['deoxys_config']
        hf.close()

        model = model_from_full_config(config, weights_file=filename)

    return model
