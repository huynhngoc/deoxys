# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..utils import Singleton


class Layers(metaclass=Singleton):
    def __init__(self):
        pass

    def register(self, layer):
        pass


def register_layer(new_layer):
    pass
