# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..utils import Singleton


class BasePreprocessor:
    def __init__(self, **kwargs):
        """
        Wrapper for creating a trained/fitted Preprocessor instance
        """
        pass

    def transform(x, y):
        return x, y


class DummyPreprocessor:
    def __init__(self):
        pass

    def transform(x, y):
        return x, y


class Preprocessors(metaclass=Singleton):
    """
    A singleton that contains all the registered customized preprocessors
    """

    def __init__(self):
        self._preprocessors = {}

    def register(self, key, preprocessor):
        if not issubclass(preprocessor, BasePreprocessor):
            raise ValueError(
                "The customized preprocessor has to be a subclass"
                + " of deoxys.data.BasePreprocessor"
            )

        if key in self._preprocessors:
            raise KeyError(
                "Duplicated key, please use another key for this preprocessor"
            )
        else:
            self._preprocessors[key] = preprocessor

    def unregister(self, key):
        if key in self._preprocessors:
            del self._preprocessors[key]

    @property
    def preprocessors(self):
        return self._preprocessors


def register_preprocessor(key, preprocessor):
    """
    Register the customized preprocessor.
    If the key name is already registered, it will raise a KeyError exception

    :param key: the unique key-name of the preprocessor
    :type key: str
    :param preprocessor: the customized preprocessor class
    :type preprocessor: deoxys.data.BasePreprocessor
    """
    Preprocessors().register(key, preprocessor)


def unregister_preprocessor(key):
    """
    Remove the registered preprocessor with the key-name

    :param key: the key-name of the preprocessor to be removed
    :type key: str
    """
    Preprocessors().unregister(key)


def _deserialize(config, custom_objects={}):
    predefined_obj = {
        'DummyPreprocessor': DummyPreprocessor
    }

    predefined_obj.update(custom_objects)

    return predefined_obj[config['class_name']](**config['config'])


def preprocessor_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define preprocessor')

    if 'config' not in config:
        # auto add empty config for preprocessor with only class_name
        config['config'] = {}
    return _deserialize(config, custom_objects=Preprocessors().preprocessors)
