# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..model.callbacks import callback_from_config


def load_train_params(train_params):
    params = dict(train_params)
    for key, val in params.items():
        if key == 'callbacks':
            if type(val) == dict:
                params[key] = [callback_from_config(val)]
            if type(val) == list:
                callbacks = []
                for v in val:
                    if type(v) == dict:
                        callbacks.append(callback_from_config(v))
                    elif type(v) == str:
                        callbacks.append(
                            callback_from_config({'class_name': v}))
                params[key] = callbacks
    return params
