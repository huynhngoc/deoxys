# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, concatenate, Lambda, \
    Add, Activation, Multiply
from tensorflow import image
import tensorflow as tf

from ..model.layers import layer_from_config
from ..utils import deep_copy


multi_input_layers = ['Add', 'AddResize', 'Concatenate', 'Multiply']
resize_input_layers = ['Concatenate', 'AddResize']


class BaseModelLoader:
    """
    The base class of all model loader. A model loader will create a
    neuralnetwork model with a predefined architecture.
    For example, UnetModelLoader creates a neural network with Unet structure.

    Raises
    ------
    NotImplementedError
        `load` method needs to be implemented in
        children classes
    """
    def load():
        raise NotImplementedError()

    def __init__(self, architecture, input_params):
        """
        Initialize a model loader

        Parameters
        ----------
        architecture : dict
            Configuration for the network architecture

            Example:
            ```
            {
                "type": "Sequential",
                "layers": [
                    {
                    "class_name": "Conv2D",
                    "config": {
                        "filters": 4,
                        "kernel_size": 3,
                        "activation": "relu",
                        "padding": "same"
                    },
                    {
                    "class_name": "Dense",
                    "config": {
                        "units": 2,
                        "activation": "sigmoid"
                    },
                ]
            }
            ```

        input_params : dict
            Configuration for the input layer

            Example
            ```
            {
                "shape": [128, 128, 3]
            }
            ```


        """
        if 'layers' in architecture:
            self._layers = deep_copy(architecture['layers'])

        self._input_params = deep_copy(input_params)


class SequentialModelLoader(BaseModelLoader):
    """
    Create a sequential network from list of layers
    """

    def load(self):
        """

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network of sequential layers
            from the configured layer list.
        """
        layers = [Input(**self._input_params)]

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)
            layers.append(next_tensor(layers[i]))
        return KerasModel(inputs=layers[0], outputs=layers[-1])


class UnetModelLoader(BaseModelLoader):
    """
    Create a unet neural network from layers

    Raises
    ------
    NotImplementedError
        volumn adjustment in skip connection
        doesn't support for 3D unet
    """

    def load(self):
        """
        Load the unet neural network.

        Example of Configuration for `layers`:
        ```
        [
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_1",
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "MaxPooling2D"
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_2",
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            ...
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_5",
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "MaxPooling2D"
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 128,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 128,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_1",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 32,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_1",
                    "conv_5"
                ]
            },
            ...
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_4",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_4",
                    "conv_2"
                ]
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_5",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 2,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_5",
                    "conv_1"
                ]
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 1,
                    "kernel_size": 1,
                    "activation": "sigmoid"
                }
            }
        ]
        ```

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with unet structure

        Raises
        ------
        NotImplementedError
            Doesn't support volumn adjustment in skip connection
        """
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)

            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if size_factors:
                        if size_factors == saved_input[
                                input_name].get_shape().as_list()[1:-1]:
                            next_input = saved_input[input_name]
                        else:
                            if len(size_factors) == 2:
                                next_input = image.resize(
                                    saved_input[input_name],
                                    size_factors,
                                    method='bilinear')
                            else:
                                raise NotImplementedError(
                                    "Resize 3D tensor not implemented")
                        inputs.append(next_input)

                    else:
                        inputs.append(saved_input[input_name])
                        size_factors = saved_input[
                            input_name].get_shape().as_list()[1:-1]
                connected_input = concatenate(inputs)
            else:
                connected_input = layers[i]

            next_layer = next_tensor(connected_input)

            if 'normalizer' in layer:
                next_layer = layer_from_config(layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[0], outputs=layers[-1])


# TODO: refactor this for Afreen
class Vnet(BaseModelLoader):
    """
    EXPERIMENTAL
    author: Afreen Mirza
    email: afreen.mirza@nmbu.no

    Create a vnet neural network from layers
    """

    def resize_by_axis(self, img, dim_1, dim_2, ax):
        resized_list = []
        # print(img.shape, ax, dim_1, dim_2)
        unstack_img_depth_list = tf.unstack(img, axis=ax)
        for j in unstack_img_depth_list:
            resized_list.append(
                image.resize(j, [dim_1, dim_2], method='bicubic'))
        stack_img = tf.stack(resized_list, axis=ax)
        # print(stack_img.shape)
        return stack_img

    def resize_along_dim(self, img, new_dim):
        dim_1, dim_2, dim_3 = new_dim

        resized_along_depth = self.resize_by_axis(img, dim_1, dim_2, 3)
        resized_along_width = self.resize_by_axis(
            resized_along_depth, dim_1, dim_3, 2)
        return resized_along_width

    def load(self):
        """
        Load the unet neural network. Use Conv3d

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with unet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        # global next_input
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)

            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if size_factors:
                        if size_factors == saved_input[
                                input_name].get_shape().as_list()[1:-1]:
                            next_input = saved_input[input_name]
                        else:
                            if len(size_factors) == 2:
                                next_input = image.resize(
                                    saved_input[input_name],
                                    size_factors,
                                    # preserve_aspect_ratio=True,
                                    method='bilinear')
                            elif len(size_factors) == 3:

                                next_input = self.resize_along_dim(
                                    saved_input[input_name],
                                    size_factors
                                )

                            else:
                                raise NotImplementedError(
                                    "Image shape is not supported ")
                        inputs.append(next_input)

                    else:
                        inputs.append(saved_input[input_name])
                        size_factors = saved_input[
                            input_name].get_shape().as_list()[1:-1]
                connected_input = concatenate(inputs)
            else:
                connected_input = layers[i]

            next_layer = next_tensor(connected_input)

            if 'normalizer' in layer:
                next_layer = layer_from_config(layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[0], outputs=layers[-1])


class DenseModelLoader(Vnet):
    def load(self):
        """
        Load the densenet neural network (2d and 3d)

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with densenet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if size_factors:
                        if size_factors == saved_input[
                                input_name].get_shape().as_list()[1:-1]:
                            next_input = saved_input[input_name]
                        else:
                            if len(size_factors) == 2:
                                next_input = image.resize(
                                    saved_input[input_name],
                                    size_factors,
                                    # preserve_aspect_ratio=True,
                                    method='bilinear')
                            elif len(size_factors) == 3:

                                next_input = self.resize_along_dim(
                                    saved_input[input_name],
                                    size_factors
                                )

                            else:
                                raise NotImplementedError(
                                    "Image shape is not supported ")
                        inputs.append(next_input)

                    else:
                        inputs.append(saved_input[input_name])
                        size_factors = saved_input[
                            input_name].get_shape().as_list()[1:-1]

                connected_input = concatenate(inputs)
            else:
                connected_input = layers[i]

            if 'dense_block' in layer:
                next_layer = self._create_dense_block(
                    layer, connected_input)
            else:
                next_tensor = layer_from_config(layer)

                next_layer = next_tensor(connected_input)

                if 'normalizer' in layer:
                    next_layer = layer_from_config(
                        layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[0], outputs=layers[-1])

    def _create_dense_block(self, layer, connected_input):
        dense = layer['dense_block']
        if type(dense) == dict:
            layer_num = dense['layer_num']
        else:
            layer_num = dense

        dense_layers = [connected_input]
        final_concat = []
        for i in range(layer_num):
            next_tensor = layer_from_config(
                {k: v for k, v in layer.items() if k != 'name'})
            if len(dense_layers) == 1:
                next_layer = next_tensor(connected_input)
            else:
                inp = concatenate(dense_layers[-2:])
                next_layer = next_tensor(inp)
                dense_layers.append(inp)

            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)
            dense_layers.append(next_layer)
            final_concat.append(next_layer)

        return concatenate(final_concat)


class ResNetModelLoader(Vnet):
    def load(self):
        """
        Load the voxresnet neural network (2d and 3d)

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with vosresnet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        layers = [Input(**self._input_params)]
        saved_input = {'input': layers[0]}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                if len(layer['inputs']) > 1:
                    inputs = []
                    size_factors = None
                    for input_name in layer['inputs']:
                        # resize based on the first input
                        if size_factors:
                            if size_factors == saved_input[
                                    input_name].get_shape().as_list()[1:-1]:
                                next_input = saved_input[input_name]
                            else:
                                if len(size_factors) == 2:
                                    next_input = image.resize(
                                        saved_input[input_name],
                                        size_factors,
                                        # preserve_aspect_ratio=True,
                                        method='bilinear')
                                elif len(size_factors) == 3:

                                    next_input = self.resize_along_dim(
                                        saved_input[input_name],
                                        size_factors
                                    )

                                else:
                                    raise NotImplementedError(
                                        "Image shape is not supported ")
                            inputs.append(next_input)

                        else:  # set resize signal
                            inputs.append(saved_input[input_name])

                            # Based on the class_name, determine resize or not
                            # No resize is required for multi-input class.
                            # Example: Add, Multiple
                            # Concatenate and Addresize requires inputs
                            # # of the same shapes.
                            # Convolutional layers with multiple inputs
                            # # have hidden concatenation, so resize is also
                            # # required.
                            layer_class = layer['class_name']
                            if layer_class in resize_input_layers or \
                                    layer_class not in multi_input_layers:
                                size_factors = saved_input[
                                    input_name].get_shape().as_list()[1:-1]

                    # No concatenation for multi-input classes
                    if layer['class_name'] in multi_input_layers:
                        connected_input = inputs
                    else:
                        connected_input = concatenate(inputs)
                else:
                    connected_input = saved_input[layer['inputs'][0]]
            else:
                connected_input = layers[i]

            # Resize back to original input
            if layer.get('resize_inputs'):
                size_factors = layers[0].get_shape().as_list()[1:-1]
                if size_factors != connected_input.get_shape().as_list()[1:-1]:
                    if len(size_factors) == 2:
                        connected_input = image.resize(
                            connected_input,
                            size_factors,
                            # preserve_aspect_ratio=True,
                            method='bilinear')
                    elif len(size_factors) == 3:
                        connected_input = self.resize_along_dim(
                            connected_input,
                            size_factors
                        )
                    else:
                        raise NotImplementedError(
                            "Image shape is not supported ")

            if 'res_block' in layer:
                next_layer = self._create_res_block(
                    layer, connected_input)
            else:
                next_tensor = layer_from_config(layer)

                next_layer = next_tensor(connected_input)

                if 'normalizer' in layer:
                    next_layer = layer_from_config(
                        layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[0], outputs=layers[-1])

    def _create_res_block(self, layer, connected_input):
        res = layer['res_block']
        if type(res) == dict:
            layer_num = res['layer_num']
        else:
            layer_num = res
        next_layer = connected_input

        for i in range(layer_num):
            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)

            if 'activation' in layer['config']:
                activation = layer['config']['activation']
                del layer['config']['activation']

                next_layer = Activation(activation)(next_layer)

            next_layer = layer_from_config(layer)(next_layer)

        return Add()([connected_input, next_layer])


###########################################################
# MultiInputModelLoader
# Experimental loader
# TODO: make this the only generic loader for Unet, ResNet, DenseNet and
# other multiple-input models
###########################################################


class MultiInputModelLoader(BaseModelLoader):  # pragma: no cover
    def resize_by_axis(self, img, dim_1, dim_2, ax):
        resized_list = []
        # print(img.shape, ax, dim_1, dim_2)
        unstack_img_depth_list = tf.unstack(img, axis=ax)
        for j in unstack_img_depth_list:
            resized_list.append(
                image.resize(j, [dim_1, dim_2], method='bicubic'))
        stack_img = tf.stack(resized_list, axis=ax)
        # print(stack_img.shape)
        return stack_img

    def resize_along_dim(self, img, new_dim):
        dim_1, dim_2, dim_3 = new_dim

        resized_along_depth = self.resize_by_axis(img, dim_1, dim_2, 3)
        resized_along_width = self.resize_by_axis(
            resized_along_depth, dim_1, dim_3, 2)
        return resized_along_width

    def _create_dense_block(self, layer, connected_input):
        dense = layer['dense_block']
        if type(dense) == dict:
            layer_num = dense['layer_num']
        else:
            layer_num = dense

        dense_layers = [connected_input]
        final_concat = []
        for i in range(layer_num):
            next_tensor = layer_from_config(
                {k: v for k, v in layer.items() if k != 'name'})
            if len(dense_layers) == 1:
                next_layer = next_tensor(connected_input)
            else:
                inp = concatenate(dense_layers[-2:])
                next_layer = next_tensor(inp)
                dense_layers.append(inp)

            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)
            dense_layers.append(next_layer)
            final_concat.append(next_layer)

        return concatenate(final_concat)

    def _create_res_block(self, layer, connected_input):
        res = layer['res_block']
        if type(res) == dict:
            layer_num = res['layer_num']
        else:
            layer_num = res
        next_layer = connected_input

        for i in range(layer_num):
            if 'normalizer' in layer:
                next_layer = layer_from_config(
                    layer['normalizer'])(next_layer)

            if 'activation' in layer['config']:
                activation = layer['config']['activation']
                del layer['config']['activation']

                next_layer = Activation(activation)(next_layer)

            next_layer = layer_from_config(
                {k: v for k, v in layer.items() if k != 'name'}
            )(next_layer)

        return Add()([connected_input, next_layer])

    def load(self):
        """
        Load the voxresnet neural network (2d and 3d)

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network with vosresnet structure

        Raises
        ------
        NotImplementedError
            Does not support video and time-series image inputs
        """
        if type(self._input_params) == dict:
            self._input_params = [self._input_params]
        num_input = len(self._input_params)
        layers = [Input(**input_params) for input_params in self._input_params]
        saved_input = {f'input_{i}': layers[i] for i in range(num_input)}

        for i, layer in enumerate(self._layers):
            if 'inputs' in layer:
                if len(layer['inputs']) > 1:
                    inputs = []
                    size_factors = None
                    for input_name in layer['inputs']:
                        # resize based on the first input
                        if size_factors:
                            if size_factors == saved_input[
                                    input_name].get_shape().as_list()[1:-1]:
                                next_input = saved_input[input_name]
                            else:
                                if len(size_factors) == 2:
                                    next_input = image.resize(
                                        saved_input[input_name],
                                        size_factors,
                                        # preserve_aspect_ratio=True,
                                        method='bilinear')
                                elif len(size_factors) == 3:

                                    next_input = self.resize_along_dim(
                                        saved_input[input_name],
                                        size_factors
                                    )

                                else:
                                    raise NotImplementedError(
                                        "Image shape is not supported ")
                            inputs.append(next_input)

                        else:  # set resize signal
                            inputs.append(saved_input[input_name])

                            # Based on the class_name, determine resize or not
                            # No resize is required for multi-input class.
                            # Example: Add, Multiple
                            # Concatenate and Addresize requires inputs
                            # # of the same shapes.
                            # Convolutional layers with multiple inputs
                            # # have hidden concatenation, so resize is also
                            # # required.
                            layer_class = layer['class_name']
                            if layer_class in resize_input_layers or \
                                    layer_class not in multi_input_layers:
                                size_factors = saved_input[
                                    input_name].get_shape().as_list()[1:-1]

                    # No concatenation for multi-input classes
                    if layer['class_name'] in multi_input_layers:
                        connected_input = inputs
                    else:
                        connected_input = concatenate(inputs)
                else:
                    connected_input = saved_input[layer['inputs'][0]]
            else:
                # independent from i, get the last layer
                connected_input = layers[-1]

            # Resize back to original input
            if layer.get('resize_inputs'):
                size_factors = layers[0].get_shape().as_list()[1:-1]
                if size_factors != connected_input.get_shape().as_list()[1:-1]:
                    if len(size_factors) == 2:
                        connected_input = image.resize(
                            connected_input,
                            size_factors,
                            # preserve_aspect_ratio=True,
                            method='bilinear')
                    elif len(size_factors) == 3:
                        connected_input = self.resize_along_dim(
                            connected_input,
                            size_factors
                        )
                    else:
                        raise NotImplementedError(
                            "Image shape is not supported ")

            if 'res_block' in layer:
                next_layer = self._create_res_block(
                    layer, connected_input)
            elif 'dense_block' in layer:
                next_layer = self._create_dense_block(
                    layer, connected_input)
            else:
                next_tensor = layer_from_config(layer)

                next_layer = next_tensor(connected_input)

                if 'normalizer' in layer:
                    next_layer = layer_from_config(
                        layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[:num_input], outputs=layers[-1])


class ModelLoaderFactory:
    _loaders = {
        'Sequential': SequentialModelLoader,
        'Unet': UnetModelLoader,
        'Vnet': Vnet,
        'Dense': DenseModelLoader,
        'DenseNet': DenseModelLoader,  # Alias
        'ResNet': ResNetModelLoader,
        'VoxResNet': ResNetModelLoader,  # Alias
        'Generic': MultiInputModelLoader,  # Alias
        'MultiInput': MultiInputModelLoader
    }

    @classmethod
    def create(cls, architecture, input_params):
        model_type = architecture['type']

        if model_type in cls._loaders:
            return cls._loaders[model_type](architecture, input_params)
        else:
            raise ValueError('Invalid Architecture {}'.format(model_type))

    @classmethod
    def register(cls, model_type, loader):
        if not issubclass(loader, BaseModelLoader):
            raise ValueError(
                "The new loader has to be a subclass"
                + " of deoxys.loader.BaseModelLoader"
            )

        if model_type in cls._loaders:
            raise KeyError(
                "Duplicated key, please use another key name for this loader"
            )
        else:
            cls._loaders[model_type] = loader


def load_architecture(architecture, input_params):
    loader = ModelLoaderFactory.create(architecture, input_params)
    return loader.load()


def register_architecture(model_type, loader):
    ModelLoaderFactory.register(model_type, loader)
