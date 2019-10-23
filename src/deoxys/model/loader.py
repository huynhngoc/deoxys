# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import model_from_config, model_from_json
from tensorflow.keras.layers import Input

from ..utils import load_data_from_json


class ModelLoader:
    @classmethod
    def create(self, json, **kwargs):
        model_type, structure = load_data_from_json(json)

        if (model_type == 'Sequential'):
            return SequentialModelLoader(structure, **kwargs)
        elif (model_type == 'Unet'):
            return UnetModelLoader(structure, **kwargs)
        elif (model_type == 'DenseModelLoader'):
            return DenseModelLoader(structure, **kwargs)
        else:
            raise ValueError()


class BaseLoader:
    def load():
        raise NotImplementedError()

    def __init__(self, structure, **kwargs):
        if 'layers' in structure:
            self._layers = structure['layers']

        if 'params' in structure:
            self._params = structure['params']

        self._input_kwargs = kwargs


class SequentialModelLoader(BaseLoader):
    """[summary]

        {
          "class_name": "Conv2D",
          "config": {
            "filters": 3,
            "kernel_size":[3, 3]
          },
          "inputs":{}
        }

    :param BaseLoader: [description]
    :type BaseLoader: [type]
    :return: [description]
    :rtype: [type]
    """

    def load(self):
        layers = [Input(**self._input_kwargs)]

        for i, layer in enumerate(self._layers):
            next_tensor = model_from_config(layer)
            layers.append(next_tensor(layers[i]))
        return KerasModel(inputs=layers[0], outputs=layers[-1]), self._params


class UnetModelLoader(BaseLoader):
    def load():
        pass


class DenseModelLoader(BaseLoader):
    def load():
        pass
