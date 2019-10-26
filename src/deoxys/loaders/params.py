# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..model.optimizers import optimizer_from_config
# from ..model.losses import loss_from_config


def load_params(model_params):
    params = dict(model_params)
    for key, val in params.items():
        if key == 'optimizer':
            if type(val) == dict:
                params[key] = optimizer_from_config(val)
        # elif key == 'loss':
        #     if type(val) == dict:
        #         model_params[key] = loss_from_config(val)

    return params