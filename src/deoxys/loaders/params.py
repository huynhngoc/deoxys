# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..model.optimizers import optimizer_from_config
from ..model.losses import loss_from_config
from ..model.metrics import metric_from_config
from ..utils import deep_copy


def load_params(model_params):
    """
    Create model objects from configurations

    :param model_params: dictionary of model objects (optimizer, loss, metrics)

    Example:
    ```
    {
        "loss": {
            "class_name": "BinaryFbetaLoss"
        },
        "optimizer": {
            "class_name": "adam",
            "config": {
                "learning_rate": 0.0001
            }
        },
        "metrics": [
            {
                "class_name": "BinaryFbeta"
            },
            "accuracy"
        ]
    }
    ```
    :type model_params: dict
    :return: [description]
    :rtype: dict
    """
    if not model_params:
        return {}
    params = deep_copy(model_params)
    for key, val in params.items():
        if key == 'optimizer':
            if type(val) == dict:
                params[key] = optimizer_from_config(val)
        elif key == 'loss':
            if type(val) == dict:
                params[key] = loss_from_config(val)
        elif key == 'metrics':
            for i, metric in enumerate(val):
                if type(metric) == dict:
                    params[key][i] = metric_from_config(metric)
                elif metric in ['TruePositives', 'FalsePositives',
                                'TrueNegatives', 'FalseNegatives']:
                    params[key][i] = metric_from_config(
                        {'class_name': metric, 'config': {'name': metric}}
                    )
    return params
