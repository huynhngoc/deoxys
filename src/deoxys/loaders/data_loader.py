# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..data.data_reader import datareader_from_config
from ..data.preprocessor import preprocessor_from_config
from ..utils import deep_copy


def load_data(dataset_params):
    """
    Create an data reader instance from a config object.

    Example:
    {
        "class_name": "HDF5Reader",
        "config": {
            "filename": "../../dataset.h5",
            "batch_size": 16,
            "x_name": "input",
            "y_name": "target",
            "batch_cache": 4,
            "train_folds": [0, 1],
            "val_folds": [2],
            "test_folds": [3, 4],
            "preprocessors": {
                "class_name": "WindowingPreprocessor",
                "config": {
                    "window_center": 1094,
                    "window_width": 200,
                    "channel": 0
                }
            }
        }
    }


    Parameters
    ----------
    dataset_params : dict
        The configuration of the data reader

    Returns
    -------
    deoxys.data.DataReader
        The data reader instance
    """
    if not dataset_params:
        return None
    params = deep_copy(dataset_params)

    if params['config']:
        for key, val in params['config'].items():
            if key == 'preprocessors':
                if type(val) == dict:
                    params['config'][key] = [preprocessor_from_config(val)]
                if type(val) == list:
                    preprocessors = []
                    for v in val:
                        if type(v) == dict:
                            preprocessors.append(preprocessor_from_config(v))
                        elif type(v) == str:
                            preprocessors.append(
                                preprocessor_from_config({'class_name': v}))
                    params['config'][key] = preprocessors

    dr = datareader_from_config(params)
    if dr.ready:
        return dr
    else:
        return None
