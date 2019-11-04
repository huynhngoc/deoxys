# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..data.data_reader import datareader_from_config


def load_data(dataset_params):
    return datareader_from_config(dataset_params)
