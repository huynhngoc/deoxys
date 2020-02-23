# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from deoxys.keras.optimizers import Adam
from deoxys.loaders import load_params
from deoxys.model.losses import BinaryFbetaLoss
from deoxys.model.metrics import BinaryFbeta
from deoxys.utils import read_file, load_json_config


def test_load_params():
    params = load_params(load_json_config(
        read_file('tests/json/model_param_config.json')))

    assert isinstance(params['optimizer'], Adam)
    assert isinstance(params['loss'], BinaryFbetaLoss)
    assert isinstance(params['metrics'][0], BinaryFbeta)
