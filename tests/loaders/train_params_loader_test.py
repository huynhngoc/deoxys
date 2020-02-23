# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from deoxys.keras.callbacks import CSVLogger, ModelCheckpoint, \
    TerminateOnNaN
from deoxys.loaders import load_train_params
from deoxys.utils import read_file, load_json_config


def test_load_params():
    params = load_train_params(load_json_config(
        read_file('tests/json/train_param_config.json')))

    assert params['epochs'] == 5
    assert len(params['callbacks']) == 3

    assert isinstance(params['callbacks'][0], CSVLogger)
    assert isinstance(params['callbacks'][1], ModelCheckpoint)
    assert isinstance(params['callbacks'][2], TerminateOnNaN)
