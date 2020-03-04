# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import tensorflow as tf

import json
import h5py
import numpy as np
from itertools import product

from ..keras.models import \
    model_from_config as keras_model_from_config, \
    model_from_json as keras_model_from_json, \
    load_model as keras_load_model, Model as KerasModel, \
    clone_model
from ..utils import is_default_tf_eager_mode

from ..keras import backend as K

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

                if weights_file:
                    # This initializes the variables used by the optimizers,
                    # as well as any stateful metric variables
                    self._model.train_on_batch(
                        *next(self.data_reader.train_generator.generate()))

                    self._model.load_weights(weights_file)
                self._compiled = True
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
                            options = {}
                            prev_layer = previous_layers(n)
                            if 'transpose' not in prev_layer:
                                options.update({'length': 300})
                            connection.append({
                                'from': prev_layer,
                                'to': layer.name,
                                **options
                            })
                    else:
                        connection.append({
                            'from': layers[inbound_layers].name,
                            'to': layer.name,
                        })
        return connection

    def activation_map(self, layer_name):
        return KerasModel(inputs=self.model.inputs,
                          outputs=self.layers[layer_name].output)

    def activation_map_for_image(self, layer_name, images):
        return self.activation_map(layer_name).predict(images, verbose=1)

    def _get_gradient_loss(self, outputs, filter_index, loss_fn):
        if loss_fn is None:
            loss_value = tf.reduce_mean(
                outputs[..., filter_index])
        else:
            loss_value = loss_fn(outputs)

        return loss_value

    def gradient_map(self, layer_name, img=None, step_size=1, epochs=20,
                     filter_index=0, loss_fn=None):
        """
        Return the gradient between the input value and the output at
        a specific node

        :param layer_name: name of the node
        :type layer_name: str
        :param img: list of input images, defaults to None. If None, a random
        image with noises will be used
        :type img: [list], optional
        :param step_size: Size of the step when performing gradient descent,
        defaults to 1
        :type step_size: int, optional
        :param epochs: Number of epochs for gradient descent, defaults to 20
        :type epochs: int, optional
        :param filter_index: index of the filter to get the gradient, can be
        any number between 0 and (size of the filters - 1), defaults to 0
        :type filter_index: int, or list, optional
        :return: list of gradient images
        :rtype: list
        """
        if type(filter_index) == int:
            list_index = [filter_index]
        else:
            list_index = filter_index

        input_shape = [1] + (self.model.input.shape)[1:]
        if img is None:
            input_img_data = np.random.random(input_shape)
        else:
            input_img_data = img

        input_img_data = [tf.Variable(
            tf.cast(input_img_data, K.floatx())) for _ in list_index]

        activation_model = self.activation_map(layer_name)

        for _ in range(epochs):
            print('epoch', _, '/', epochs)

            for i, filter_index in enumerate(list_index):
                print('filter', filter_index)
                if is_default_tf_eager_mode():

                    with tf.GradientTape() as tape:
                        outputs = activation_model(input_img_data[i])

                        loss_value = self._get_gradient_loss(
                            outputs, filter_index, loss_fn)
                        # if loss_fn is None:
                        #     loss_value = tf.reduce_mean(
                        #         outputs[..., filter_index])
                        # else:
                        #     loss_value = loss_fn(outputs)

                    grads = tape.gradient(loss_value, input_img_data[i])
                else:
                    outputs = activation_model.output

                    loss_value = self._get_gradient_loss(
                        outputs, filter_index, loss_fn)

                    # if loss_fn is None:
                    #     loss_value = tf.reduce_mean(
                    #         outputs[..., filter_index])
                    # else:
                    #     loss_value = loss_fn(outputs)

                    gradient = K.gradients(loss_value, activation_model.input)

                    grads = K.function(activation_model.input,
                                       gradient)(input_img_data)[0]

                normalized_grads = grads / \
                    (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

                input_img_data[i].assign_add(normalized_grads * step_size)

        if len(input_img_data) > 1:
            if is_default_tf_eager_mode():
                return [K.get_value(input_img) for input_img in input_img_data]
            else:
                return [K.eval(input_img) for input_img in input_img_data]

        if is_default_tf_eager_mode():
            return K.get_value(input_img_data[0])
        else:
            return K.eval(input_img_data[0])

    def gradient_map_generator(self, layer_name, img=None, step_size=1,
                               filter_index=0, loss_fn=None):
        """
        EXPERIMENTAL

        :param layer_name: [description]
        :type layer_name: [type]
        :param img: [description], defaults to None
        :type img: [type], optional
        :param step_size: [description], defaults to 1
        :type step_size: int, optional
        :param filter_index: [description], defaults to 0
        :type filter_index: int, optional
        :param loss_fn: [description], defaults to None
        :type loss_fn: [type], optional
        :yield: [description]
        :rtype: [type]
        """
        if type(filter_index) == int:
            list_index = [filter_index]
        else:
            list_index = filter_index

        input_shape = [1] + (self.model.input.shape)[1:]
        if img is None:
            input_img_data = np.random.random(input_shape)
        else:
            input_img_data = img

        input_img_data = [tf.Variable(
            tf.cast(input_img_data, K.floatx())) for _ in list_index]

        activation_model = self.activation_map(layer_name)

        w, h = input_shape[1:3]
        l, r = int(w / 2 - 5), int(w / 2 + 5)
        u, d = int(h / 2 - 5), int(h / 2 + 5)

        e = 0
        while True:
            print('epoch:', e)
            e += 1
            cont = False
            if l > 0:
                if e % 3 == 0:
                    l -= 2  # noqa
                cont = True
            if r < w:
                if e % 3 == 0:
                    r += 2
                cont = True
            if u > 0:
                if e % 3 == 0:
                    u -= 2
                cont = True
            if d < h:
                if e % 3 == 0:
                    d += 2
                cont = True
            horizon = [
                (max(0, l - w // 4), r - w // 4),
                (l + w // 4, min(w, r + w // 4)),
                (l, r)
            ]
            vertical = [
                (max(0, u - h // 4), d - h // 4),
                (u + h // 4, min(h, d + h // 4)),
                (u, d)
            ]

            for i, filter_index in enumerate(list_index):
                print('filter', filter_index)
                if is_default_tf_eager_mode():

                    with tf.GradientTape() as tape:
                        outputs = activation_model(input_img_data[i])
                        if loss_fn is None:
                            loss_value = tf.reduce_mean(
                                outputs[..., filter_index])
                        else:
                            loss_value = loss_fn(outputs)

                    grads = tape.gradient(loss_value, input_img_data[i])
                else:
                    outputs = activation_model.output

                    if loss_fn is None:
                        loss_value = tf.reduce_mean(
                            outputs[..., filter_index])
                    else:
                        loss_value = loss_fn(outputs)

                    gradient = K.gradients(loss_value, activation_model.input)

                    grads = K.function(activation_model.input,
                                       gradient)(input_img_data)[0]
                normalized_grads = grads / \
                    (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

                input_img_data[i].assign_add(normalized_grads * step_size)

                if cont:
                    modified = np.random.random(
                        input_shape) - K.get_value(input_img_data[i])
                    # modified[:, l:r, u:d, :]=0
                    for (left, right), (up, down) in product(
                            horizon, vertical):
                        modified[:, left:right, up:down, :] = 0

                    input_img_data[i].assign_add(modified)
                # else:
                #     res = K.get_value(input_img_data[i])
                #     center = res[:, int(w/4): int(3*w/4),
                #                  int(h/4): int(3*h/4), :]
                #     input_img_data[i] = tf.image.resize(
                #         center, (w, h), method='bilinear')

            if len(input_img_data) > 1:
                if is_default_tf_eager_mode():
                    yield [K.get_value(input_img)
                           for input_img in input_img_data]
                else:
                    yield [K.eval(input_img) for input_img in input_img_data]
            else:
                if is_default_tf_eager_mode():
                    yield K.get_value(input_img_data[0])
                else:
                    yield K.eval(input_img_data[0])

    def _get_backprop_loss(self, output, mode='max', output_index=0):
        if mode == 'max':
            loss = K.max(output, axis=-1)
        elif mode == 'one':
            loss = output[..., output_index]
        else:
            loss = output

        return loss

    def _max_filter_map(self, output):
        return K.argmax(output, axis=-1)

    def _backprop_eagerly(self, layer_name, images, mode='max',
                          output_index=0):
        img_tensor = tf.Variable(tf.cast(images, K.floatx()))
        activation_map = self.activation_map(layer_name)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            output = activation_map(img_tensor)

            loss = self._get_backprop_loss(output, mode, output_index)
            # if mode == 'max':
            #     loss = K.max(output, axis=3)
            # elif mode == 'one':
            #     loss = output[..., output_index]
            # else:
            #     loss = output

        grads = tape.gradient(loss, img_tensor)

        return K.get_value(grads)

    def _backprop_symbolic(self, layer_name, images, mode='max',
                           output_index=0):
        output = self.layers[layer_name].output

        loss = self._get_backprop_loss(output, mode, output_index)
        # if mode == 'max':
        #     loss = K.max(output, axis=3)
        # elif mode == 'one':
        #     loss = output[..., output_index]
        # else:
        #     loss = output

        grads = K.gradients(loss, self.model.input)[0]

        fn = K.function(self.model.input, grads)

        return fn(images)

    def backprop(self, layer_name, images, mode='max', output_index=0):
        if is_default_tf_eager_mode():
            grads = self._backprop_eagerly(
                layer_name, images, mode, output_index)
        else:
            grads = self._backprop_symbolic(
                layer_name, images, mode, output_index)
        return grads

    def _gradient_backprop(self, gradient_name, layer_name,
                           images, mode, output_index):
        # save current weight
        weights = self.model.get_weights()

        with tf.Graph().as_default() as g:
            with g.gradient_override_map({'Relu': gradient_name}):
                tf.compat.v1.experimental.output_all_intermediates(True)
                new_model = clone_model(self.model)
                # Apply weights
                new_model.set_weights(weights)

                if mode == 'max':
                    output = K.max(new_model.get_layer(
                        layer_name).output, axis=3)
                elif mode == 'one':
                    output = new_model.get_layer(
                        layer_name).output[..., output_index]
                else:
                    output = new_model.get_layer(
                        layer_name).output

                grads = K.gradients(output, new_model.input)[0]

                fn = K.function(new_model.input, grads)

                grad_output = fn(images)

            del new_model
        del g

        return grad_output

    def deconv(self, layer_name, images, mode='max',
               output_index=0):
        return self._gradient_backprop('DeconvNet', layer_name,
                                       images, mode, output_index)

    def guided_backprop(self, layer_name, images, mode='max',
                        output_index=0):
        return self._gradient_backprop('GuidedBackProp', layer_name,
                                       images, mode, output_index)

    def max_filter(self, layer_name, images):
        return self._max_filter_map(
            self.activation_map_for_image(layer_name, images))

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


def model_from_full_config(model_config, weights_file=None, **kwargs):
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
        weights_file=weights_file,
        **kwargs)


def model_from_config(architecture, input_params,
                      model_params=None, train_params=None,
                      dataset_params=None, weights_file=None, **kwargs):
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
                 data_generator, config=config, weights_file=weights_file,
                 **kwargs)


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

        with h5py.File(filename) as hf:
            if 'deoxys_config' in hf.attrs.keys():
                config = hf.attrs['deoxys_config']
                model._data_reader = load_data(config['dataset_params'])

    except Exception:
        hf = h5py.File(filename, 'r')
        if 'deoxys_config' in hf.attrs.keys():
            config = hf.attrs['deoxys_config']
        hf.close()

        model = model_from_full_config(config, weights_file=filename)

    return model


@tf.RegisterGradient("GuidedBackProp")
def _GuidedBackProp(op, grad):
    dtype = op.inputs[0].dtype
    return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)


@tf.RegisterGradient("DeconvNet")
def _DeconvNet(op, grad):
    dtype = op.inputs[0].dtype
    return grad * tf.cast(grad > 0., dtype)


@tf.RegisterGradient("BackProp")
def _BackProp(op, grad):
    dtype = op.inputs[0].dtype
    return grad * tf.cast(op.inputs[0] > 0., dtype)
