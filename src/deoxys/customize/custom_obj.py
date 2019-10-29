# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..utils import Singleton


class CustomObj(metaclass=Singleton):
    """
    A singleton that contains all the registered customized obj
    """

    def __init__(self):
        self._obj = {}

    def register(self, key, obj):
        if key in self._obj:
            raise KeyError(
                "Duplicated key, please use another key name for this obj"
            )
        else:
            self._obj[key] = obj

    def unregister(self, key):
        if key in self._obj:
            del self._obj[key]

    @property
    def obj(self):
        return self._obj


def register_obj(key, obj):
    CustomObj().register(key, obj)


def unregister_obj(key):
    """
    Remove the registered obj with the key-name

    :param key: the key-name of the obj to be removed
    :type key: str
    """
    CustomObj().unregister(key)
