# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..data.data_reader import datareader_from_config
from ..data.preprocessor import preprocessor_from_config


def load_data(dataset_params):
    if not dataset_params:
        return {}
    params = dict(dataset_params)

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

    return datareader_from_config(params)
