# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from deoxys.keras.models import Model as KerasModel, mode as keras_mode
from deoxys.keras.layers import Input, concatenate, Lambda
from tensorflow import image
import tensorflow as tf

from ..model.layers import layer_from_config
from ..utils import deep_copy


class BaseModelLoader:
    """
    The base class of all model loader. A model loader will create a
    neuralnetwork model with a predefined architecture.
    For example, UnetModelLoader creates a neural network with Unet structure.

    :raises NotImplementedError: `load` method needs to be implemented in
    children classes
    """
    def load():
        raise NotImplementedError()

    def __init__(self, architecture, input_params):
        """
        Initialize a model loader.

        :param architecture: configuration for the network architecture
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
        :type architecture: dict
        :param input_params: configuration for the input layer
        ```
        {
            "shape": [128, 128, 3]
        }
        ```
        :type input_params: dict
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
        :return: A neural network of sequential layers
        from the configured layer list.
        :rtype: tensorflow.keras.models.Model
        """
        layers = [Input(**self._input_params)]

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)
            layers.append(next_tensor(layers[i]))
        return KerasModel(inputs=layers[0], outputs=layers[-1])


class UnetModelLoader(BaseModelLoader):
    """
    Create a unet neural network from layers

    :raises NotImplementedError: volumn adjustment in skip connection
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

        :raises NotImplementedError: volumn adjustment in skip connection
        doesn't support
        :return: A neural network with unet structure
        :rtype: tensorflow.keras.models.Model
        """
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)

            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if keras_mode.upper() == 'ALONE':
                        # keras issue: convtranspose layer output shape are
                        # (None, None, None, filters)
                        if saved_input[
                                input_name].get_shape() != saved_input[
                                    input_name]._keras_shape:
                            saved_input[input_name].set_shape(
                                saved_input[input_name]._keras_shape)
                    if size_factors:
                        if size_factors == saved_input[
                                input_name].get_shape().as_list()[1:-1]:
                            next_input = saved_input[input_name]
                        else:
                            if len(size_factors) == 2:
                                if keras_mode.upper() == 'ALONE':
                                    next_input = Lambda(
                                        lambda input_tensor: image.resize(
                                            input_tensor,
                                            size_factors,
                                            # preserve_aspect_ratio=True,
                                            method='bilinear')
                                    )(saved_input[input_name])
                                else:
                                    next_input = image.resize(
                                        saved_input[input_name],
                                        size_factors,
                                        # preserve_aspect_ratio=True,
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
    Create a unet neural network from layers

    :raises NotImplementedError: volumn adjustment in skip connection
    doesn't support for 3D unet
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

        :raises NotImplementedError: volumn adjustment in skip connection
        doesn't support
        :return: A neural network with unet structure
        :rtype: tensorflow.keras.models.Model
        """
        global next_input
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
        print(layers[0], layers[-1])
        return KerasModel(inputs=layers[0], outputs=layers[-1])


class DenseModelLoader(BaseModelLoader):
    def load():
        pass


class ModelLoaderFactory:
    _loaders = {
        'Sequential': SequentialModelLoader,
        'Unet': UnetModelLoader,
        'Vnet': Vnet,
        'Dense': DenseModelLoader
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
