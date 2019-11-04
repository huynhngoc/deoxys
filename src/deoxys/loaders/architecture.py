# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, concatenate
from tensorflow import image

from ..model.layers import layer_from_config


class BaseModelLoader:
    def load():
        raise NotImplementedError()

    def __init__(self, architecture, input_params):
        if 'layers' in architecture:
            self._layers = architecture['layers']

        self._input_params = input_params


class SequentialModelLoader(BaseModelLoader):
    """[summary]

        {
          "class_name": "Conv2D",
          "config": {
            "filters": 3,
            "kernel_size":[3, 3]
          }
        }

    :param BaseLoader: [description]
    :type BaseLoader: [type]
    :return: [description]
    :rtype: [type]
    """

    def load(self):
        layers = [Input(**self._input_params)]

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)
            layers.append(next_tensor(layers[i]))
        return KerasModel(inputs=layers[0], outputs=layers[-1])


class UnetModelLoader(BaseModelLoader):
    def load(self):
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)

            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if size_factors:
                        if len(size_factors) == 2:
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


class DenseModelLoader(BaseModelLoader):
    def load():
        pass


class ModelLoaderFactory:
    _loaders = {
        'Sequential': SequentialModelLoader,
        'Unet': UnetModelLoader,
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
