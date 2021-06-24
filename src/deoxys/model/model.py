# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


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
from ..utils import is_default_tf_eager_mode, number_of_iteration

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


    Parameters
    ----------
    model : tensorflow.keras.models.Model
        a keras model object
    model_params : dict, optional
        params to compile a keras model, by default None
    train_params : dict, optional
        params for training, evaluate, predict, by default None
    data_reader : deoxys.data.DataReader, optional
        A deoxys data reader, by default None
    pre_compiled : bool, optional
        True if model has been compiled, by default False
    weights_file : str, optional
        path to h5 file that contains the weights of the
        keras model, by default None
    config : dict, optional
        full config to create the model, by default None

    Raises
    ------
    ValueError
        raises error if model_params is defined without optimizer
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
                 pre_compiled=False, weights_file=None, config=None,
                 sample_data=None):
        """
        Create a deoxys model
        """

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
                    if self._data_reader is not None:
                        batch_x, batch_y = next(
                            self.data_reader.train_generator.generate())
                    elif sample_data is not None:
                        batch_x, batch_y = sample_data
                    else:
                        shape_x = self._model.input_shape[1:]
                        shape_y = self.model.output_shape[1:]
                        try:
                            batch_x = np.zeros((1, *shape_x))
                            batch_y = np.zeros((1, *shape_y))
                        except TypeError:
                            dim_x = (64,)*(len(shape_x) - 1)
                            dim_y = (64,)*(len(shape_y) - 1)
                            batch_x = np.zeros((1, *dim_x, shape_x[-1]))
                            batch_y = np.zeros((1, *dim_y, shape_y[-1]))

                    self._model.train_on_batch(
                        batch_x, batch_y
                    )

                    self._model.load_weights(weights_file)
                self._compiled = True
            else:
                raise ValueError('optimizer is a required parameter in '
                                 'model_params.')

    def compile(self, optimizer=None, loss=None, metrics=None,
                loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None,
                target_tensors=None, **kwargs):
        """
        Raises
        ------
        Warning
            calling this function will recompile the model with
            new configuration
        """
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

        Parameters
        ----------
        filename : str
            name of the file
        """
        self._model.save(filename, *args, **kwargs)

        if self.config:
            config = json.dumps(self.config)

            with h5py.File(filename, 'a') as saved_model:
                saved_model.attrs.create('deoxys_config', config)
                if self._data_reader is not None:
                    batch_x, batch_y = next(
                        self.data_reader.train_generator.generate())
                    group = saved_model.create_group('deoxys')
                    group.create_dataset('batch_x', data=batch_x)
                    group.create_dataset('batch_y', data=batch_y)

    def fit(self, *args, **kwargs):
        """
        Train model
        """
        # Reset layer map as weight will change after training
        self._layers = None

        self._model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict model
        """
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate model
        """
        return self._model.evaluate(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        # Reset layer map as weight will change after training
        self._layers = None
        return self._model.fit_generator(*args, **kwargs)

    def evaluate_generator(self, *args, **kwargs):
        return self._model.evaluate(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def fit_train(self, **kwargs):
        """
        Train the model with training data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        # Reset layer map as weight will change after training
        self._layers = None

        params = self._get_train_params(self._fit_param_keys, **kwargs)
        train_data_gen = self._data_reader.train_generator

        if number_of_iteration():
            train_steps_per_epoch = number_of_iteration()
        else:
            train_steps_per_epoch = train_data_gen.total_batch

        val_data_gen = self._data_reader.val_generator
        val_steps_per_epoch = val_data_gen.total_batch

        return self.fit_generator(train_data_gen.generate(),
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_data_gen.generate(),
                                  validation_steps=val_steps_per_epoch,
                                  **params)

    def evaluate_train(self, **kwargs):  # pragma: no cover
        """
        Evaluate the model on the training data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.train_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate(data_gen.generate(),
                             steps=steps_per_epoch,
                             **params)

    def evaluate_val(self, **kwargs):  # pragma: no cover
        """
        Evaluate model's performance using validation data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.val_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate(data_gen.generate(),
                             steps=steps_per_epoch,
                             **params)

    def predict_val(self, **kwargs):
        """
        Predict validation data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.val_generator
        steps_per_epoch = data_gen.total_batch

        return self.predict(data_gen.generate(),
                            steps=steps_per_epoch,
                            **params)

    def predict_val_generator(self, **kwargs):  # pragma: no cover
        """
        Predict validation data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.val_generator
        total_batch = data_gen.total_batch
        for i, (x,) in enumerate(data_gen.generate()):
            if i == total_batch:
                break
            yield self.predict(x,
                               **params)

    def evaluate_test(self, **kwargs):
        """
        Evaluate model performance using test data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._evaluate_param_keys, **kwargs)
        data_gen = self._data_reader.test_generator
        steps_per_epoch = data_gen.total_batch

        return self.evaluate(data_gen.generate(),
                             steps=steps_per_epoch,
                             **params)

    def predict_test(self, **kwargs):
        """
        Predict test data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.test_generator
        steps_per_epoch = data_gen.total_batch

        return self.predict(data_gen.generate(),
                            steps=steps_per_epoch,
                            **params)

    def predict_test_generator(self, **kwargs):  # pragma: no cover
        """
        Predict test data
        """
        if self._data_reader is None:
            raise Warning('No DataReader is specified. This action is ignored')
            return None

        params = self._get_train_params(self._predict_param_keys, **kwargs)
        data_gen = self._data_reader.test_generator
        total_batch = data_gen.total_batch

        for i, (x,) in enumerate(data_gen.generate()):
            if i == total_batch:
                break
            yield self.predict(x,
                               **params)

    @property
    def is_compiled(self):
        return self._compiled

    @property
    def model(self):
        """
        Return the keras model
        """
        return self._model

    @property
    def data_reader(self):
        """
        Get the data reader used in this model
        """
        return self._data_reader

    @property
    def layers(self):
        """
        Get the dictionary of layers in the model

        Returns
        -------
        dict
            dictionary of layers
        """
        if self._layers is None:
            self._layers = {layer.name: layer for layer in self.model.layers}

        return self._layers

    @property
    def node_graph(self):  # pragma: no cover
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

    def sub_model(self, layer_name):
        """
        Create a sub-model with the same inputs, and the outputs of a specific
        layer in the deoxys model.

        Parameters
        ----------
        layer_name : str
            name of layer

        Returns
        -------
        tensorflow.keras.models.Model
            Model, whose outputs are of the layer_name
        """
        return KerasModel(inputs=self.model.inputs,
                          outputs=self.layers[layer_name].output)

    def activation_map(self, layer_name, images):
        """
        Get activation map of a list of images
        """
        model = self.sub_model(layer_name)
        res = model.predict(images, verbose=1)

        del model
        return res

    def _get_gradient_loss(self, outputs, filter_index, loss_fn):
        if loss_fn is None:
            loss_value = tf.reduce_mean(
                outputs[..., filter_index])
        else:
            loss_value = loss_fn(outputs)

        return loss_value

    def activation_maximization(self, layer_name, img=None,
                                step_size=1, epochs=20,
                                filter_index=0, loss_fn=None,
                                verbose=True):
        """
        Return the image that maximize the activation output of one or more
        filters in a specific layer.

        Parameters
        ----------
        layer_name: str
            name of the node
        img: [type], optional
            list of initial images, by default None

            If None, a random
            image with noises will be used.

        step_size: int, optional
            Size of the step when performing gradient descent, by default 1
        epochs: int, optional
            Number of epochs for gradient descent, by default 20
        filter_index: int, or list, optional
            index of the filter to get the gradient, can be
            any number between 0 and (size of the filters - 1), by default 0
        loss_fn: callable, optional
            customized loss function, by default None
        verbose: bool, optional
            By default True

        Returns
        -------
        list
            list of images that maximize the activation's filters
        """
        if type(filter_index) == int:
            list_index = [filter_index]
        else:
            list_index = filter_index

        input_shape = [1] + list((self.model.input.shape)[1:])
        if img is None:
            input_img_data = np.random.random(input_shape)
        else:
            input_img_data = img

        input_img_data = [tf.Variable(
            tf.cast(input_img_data, K.floatx())) for _ in list_index]

        activation_model = self.sub_model(layer_name)

        for _ in range(epochs):
            if verbose:
                print('epoch', _, '/', epochs)

            for i, filter_index in enumerate(list_index):
                if verbose:
                    print('filter', filter_index)
                if is_default_tf_eager_mode():

                    with tf.GradientTape() as tape:
                        outputs = activation_model(input_img_data[i])

                        loss_value = self._get_gradient_loss(
                            outputs, filter_index, loss_fn)

                    grads = tape.gradient(loss_value, input_img_data[i])
                else:  # pragma: no cover
                    outputs = activation_model.output

                    loss_value = self._get_gradient_loss(
                        outputs, filter_index, loss_fn)

                    gradient = K.gradients(loss_value, activation_model.input)

                    grads = K.function(activation_model.input,
                                       gradient)(input_img_data)[0]

                normalized_grads = grads / \
                    (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

                input_img_data[i].assign_add(normalized_grads * step_size)

        if len(input_img_data) > 1:
            if is_default_tf_eager_mode():
                return [K.get_value(input_img) for input_img in input_img_data]
            else:  # pragma: no cover
                return [K.eval(input_img) for input_img in input_img_data]

        if is_default_tf_eager_mode():
            return K.get_value(input_img_data[0])
        else:  # pragma: no cover
            return K.eval(input_img_data[0])

    def _get_backprop_loss(self, output, mode='max', output_index=0,
                           loss_fn=None):
        if mode == 'max':
            loss = K.max(output, axis=-1)
        elif mode == 'mean':
            loss = K.mean(output, axis=-1)
        elif mode == 'min':
            loss = K.min(output, axis=-1)
        elif mode == 'one':
            loss = output[..., output_index]
        elif mode == 'all':
            loss = output
        elif loss_fn is not None:
            loss = loss_fn(output)
        else:
            loss = output

        return loss

    def _max_filter_map(self, output):
        return K.argmax(output, axis=-1)

    def _backprop_eagerly(self, layer_name, images, mode='max',
                          output_index=0, loss_fn=None):
        img_tensor = tf.Variable(tf.cast(images, K.floatx()))
        activation_model = self.sub_model(layer_name)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            output = activation_model(img_tensor)

            loss = self._get_backprop_loss(output, mode, output_index, loss_fn)

        grads = tape.gradient(loss, img_tensor)

        return K.get_value(grads)

    def _backprop_symbolic(self, layer_name, images, mode='max',
                           output_index=0, loss_fn=None):  # pragma: no cover
        output = self.layers[layer_name].output

        loss = self._get_backprop_loss(output, mode, output_index, loss_fn)

        grads = K.gradients(loss, self.model.input)[0]

        fn = K.function(self.model.input, grads)

        return fn(images)

    def backprop(self, layer_name, images, mode='max', output_index=0,
                 loss_fn=None):
        """
        Return saliency map, or backprop, or gradient map of a list of images

        Parameters
        ----------
        layer_name : str
            name of the layer
        images : list
            list of images
        mode : str, optional
            mode to calculate the loss to use when backpropagation,
            by default 'max', other options are 'mean', 'min', 'one', 'custom',
            and 'all'.

            'max', 'min', 'mean': calculate the loss by calculating the
            max, min, or mean over the inner most axis(axis=-1) of the output.

            'one': calculate the loss on one index in the inner-most
            axis(axis=-1) of the output of the layer.

            'all': use the output of the layer as the loss score.

            'custom': use a custom function to calculate the loss score based
            on the output of the layer.

        output_index : int, optional
            use when mode = 'one', by default 0
        loss_fn : callable, optional
            use when mode = 'custom', the function to calculate the
            loss score based on the output of the layer, by default None

        Returns
        -------
        numpy.array of images
            resulting images when performing backpropagation
        """
        if is_default_tf_eager_mode():
            grads = self._backprop_eagerly(
                layer_name, images, mode, output_index, loss_fn)
        else:  # pragma: no cover
            grads = self._backprop_symbolic(
                layer_name, images, mode, output_index, loss_fn)
        return grads

    def _gradient_backprop(self, gradient_name, layer_name,
                           images, mode, output_index,
                           loss_fn=None):  # pragma: no cover
        # save current weight
        weights = self.model.get_weights()

        with tf.Graph().as_default() as g:
            with g.gradient_override_map({'Relu': gradient_name}):
                tf.compat.v1.experimental.output_all_intermediates(True)
                new_model = clone_model(self.model)
                # Apply weights
                new_model.set_weights(weights)

                output = self._get_backprop_loss(
                    new_model.get_layer(layer_name).output,
                    mode, output_index, loss_fn)

                # if mode == 'max':
                #     output = K.max(new_model.get_layer(
                #         layer_name).output, axis=-1)
                # elif mode == 'one':
                #     output = new_model.get_layer(
                #         layer_name).output[..., output_index]
                # elif mode == 'all':
                #     output = new_model.get_layer(
                #         layer_name).output
                # elif loss_fn is not None:
                #     output = loss_fn(new_model.get_layer(layer_name).output)
                # else:
                #     output = new_model.get_layer(
                #         layer_name).output

                grads = K.gradients(output, new_model.input)[0]

                fn = K.function(new_model.input, grads)

                grad_output = fn(images)

            del new_model
        del g

        return grad_output

    def deconv(self, layer_name, images, mode='max',
               output_index=0, loss_fn=None):
        if is_default_tf_eager_mode():
            return self._gradient_backprop_eager(
                _DeconvRelu, layer_name,
                images, mode, output_index, loss_fn
            )
        else:  # pragma: no cover
            return self._gradient_backprop('DeconvNet', layer_name,
                                           images, mode, output_index, loss_fn)

    def guided_backprop(self, layer_name, images, mode='max',
                        output_index=0, loss_fn=None):
        if is_default_tf_eager_mode():
            return self._gradient_backprop_eager(
                _GuidedBackPropRelu, layer_name,
                images, mode, output_index, loss_fn
            )
        else:  # pragma: no cover
            return self._gradient_backprop('GuidedBackProp', layer_name,
                                           images, mode, output_index, loss_fn)

    def _gradient_backprop_eager(self, grad_fn, layer_name, images, mode='max',
                                 output_index=0, loss_fn=None):
        # save current weight
        weights = self.model.get_weights()

        new_model = clone_model(self.model)
        # Apply weights
        new_model.set_weights(weights)

        for layer in new_model.layers:
            if 'activation' in layer.get_config():
                if 'relu' in layer.activation.__name__:
                    layer.activation = grad_fn

        guided_model = KerasModel(new_model.inputs,
                                  new_model.get_layer(layer_name).output)

        img_tensor = tf.Variable(tf.cast(images, K.floatx()))
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            output = guided_model(img_tensor)

            loss = self._get_backprop_loss(output, mode, output_index, loss_fn)

        grads = tape.gradient(loss, img_tensor)

        del guided_model
        del new_model

        return K.get_value(grads)

    def max_filter(self, layer_name, images):
        """
        Return a list of images in which each pixel value is the index of the
        filter having the max value in the activation map.
        """
        return self._max_filter_map(
            self.activation_map(layer_name, images))

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
    """[summary]

    Parameters
    ----------
    model_config : str or dict
        a JSON string or a dictionary contains the
        architecture, model_params, input_params configuration of the model
    weights_file : str, optional
        path to the saved weight file, by default None

    Returns
    -------
    deoxys.model.Model
        The model

    Raises
    ------
    ValueError
        When architecture or input_params are missing
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
                      dataset_params=None, weights_file=None, sample_data=None,
                      **kwargs):
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
                 sample_data=sample_data,
                 **kwargs)


def model_from_keras_config(config, **kwarg):
    return Model(keras_model_from_config(config), **kwarg)


def model_from_keras_json(json, **kwarg):
    return Model(keras_model_from_json(json), **kwarg)


def load_model(filename, **kwargs):
    """
    Load model from file

    Parameters
    ----------
    filename : str
        path to the h5 file

    Returns
    -------
    deoxys.model.Model
        The loaded model
    """
    # Keras got the error of loading custom object
    try:
        loaded_model = keras_load_model(filename,
                                        custom_objects={
                                            **Layers().layers,
                                            **Activations().activations,
                                            **Losses().losses,
                                            **Metrics().metrics})

        # keyword arguments to create the model
        model_kwargs = {}
        with h5py.File(filename, 'r') as hf:
            # get the data_reader
            if 'deoxys_config' in hf.attrs.keys():
                config = hf.attrs['deoxys_config']
                config = load_json_config(config)

                if 'dataset_params' in config:
                    model_kwargs['data_reader'] = load_data(
                        config['dataset_params'])

            # take the sample data
            if 'deoxys' in hf.keys():
                if 'batch_x' in hf['deoxys'] and 'batch_y' in hf['deoxys']:
                    model_kwargs['sample_data'] = (hf['deoxys']['batch_x'][:],
                                                   hf['deoxys']['batch_y'][:])

            # User input will overwrites all existing args
            model_kwargs.update(kwargs)

        model = Model(loaded_model, pre_compiled=True, **model_kwargs)

    except Exception:
        sample_data = None
        with h5py.File(filename, 'r') as hf:
            if 'deoxys_config' in hf.attrs.keys():
                config = hf.attrs['deoxys_config']

            if 'deoxys' in hf.keys():
                if 'batch_x' in hf['deoxys'] and 'batch_y' in hf['deoxys']:
                    sample_data = (hf['deoxys']['batch_x'][:],
                                   hf['deoxys']['batch_y'][:])

        model = model_from_full_config(
            config, weights_file=filename, sample_data=sample_data)

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


@tf.custom_gradient
def _GuidedBackPropRelu(x):
    res = K.relu(x)

    def grad(dy):
        return dy * tf.cast(dy > 0., x.dtype) * tf.cast(x > 0., x.dtype)

    return res, grad


@tf.custom_gradient
def _DeconvRelu(x):
    res = K.relu(x)

    def grad(dy):
        return dy * tf.cast(dy > 0., x.dtype)

    return res, grad


@tf.custom_gradient
def _BackPropRelu(x):
    res = K.relu(x)

    def grad(dy):
        return dy * tf.cast(x > 0., x.dtype)

    return res, grad
