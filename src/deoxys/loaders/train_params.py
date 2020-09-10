# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..model.callbacks import callback_from_config
from ..utils import deep_copy


def load_train_params(train_params):
    """
    Create training parameters from configuration

    Parameters
    ----------
    train_params : dict
        dictionary of training parameters configuration (epochs, callbacks)

        Example:
        ```
        {
            "epoch": 10,
            "callbacks": [
                "CSVLogger",
                {
                    "class_name": "ModelCheckpoint"
                }
            ]
        }
        ```

    Returns
    -------
    dict
        Dictionary of training parameters objects
    """
    if not train_params:
        return {}
    params = deep_copy(train_params)
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
